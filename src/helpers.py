from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_column(x):
    if isinstance(x, list):
        return str(x)
    elif pd.isna(x):
        return "Unknown"
    return str(x)


class TemporalOutlierDetector(BaseEstimator, TransformerMixin):
    def __init__(self, time_column='timestamp', window_size='D', group_column='parent_asin'):
        self.time_column = time_column
        self.window_size = window_size
        self.group_column = group_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['datetime'] = pd.to_datetime(X[self.time_column], unit='s', errors='coerce')
        X['review_count'] = X.groupby([self.group_column, X['datetime'].dt.to_period(self.window_size)])['datetime'].transform('count')
        X['temporal_outlier'] = (X['review_count'] > X.groupby(self.group_column)['review_count'].transform(lambda x: x.quantile(0.95))).astype(int)
        return X[['temporal_outlier']].values


class BehavioralOutlierDetector(BaseEstimator, TransformerMixin):
    def __init__(self, user_column='user_id', time_column='timestamp', rating_column='rating', window_size='D', review_threshold=3, rating_deviation_threshold=1.5, group_column='parent_asin'):
        self.user_column = user_column
        self.time_column = time_column
        self.rating_column = rating_column
        self.window_size = window_size
        self.review_threshold = review_threshold
        self.rating_deviation_threshold = rating_deviation_threshold
        self.group_column = group_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['datetime'] = pd.to_datetime(X[self.time_column], errors='coerce')

        # High-frequency user detection
        user_review_counts = X.groupby([self.group_column, self.user_column, X['datetime'].dt.to_period(self.window_size)]).size().reset_index(name='review_count')
        high_frequency_users = user_review_counts[user_review_counts['review_count'] > self.review_threshold][self.user_column].unique()

        # Rating deviation detection
        user_avg_ratings = X.groupby([self.group_column, self.user_column])[self.rating_column].mean().reset_index()
        overall_avg_ratings = X.groupby(self.group_column)[self.rating_column].mean().reset_index()
        deviating_users = user_avg_ratings.merge(overall_avg_ratings, on=self.group_column, suffixes=('_user', '_overall'))
        deviating_users['rating_deviation'] = abs(deviating_users[f'{self.rating_column}_user'] - deviating_users[f'{self.rating_column}_overall'])
        deviating_users = deviating_users[deviating_users['rating_deviation'] > self.rating_deviation_threshold][self.user_column].unique()

        # Combine both outlier types into a single score
        X['behavioral_outlier'] = ((X[self.user_column].isin(high_frequency_users) | 
                                    X[self.user_column].isin(deviating_users))).astype(int)

        return X[['behavioral_outlier']].values

class GroupwiseIsolationForest(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.1, random_state=42, max_samples=1000, n_estimators=100):
        self.contamination = contamination
        self.random_state = random_state
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.group_models = {}

    def fit(self, X, y=None):
        groups = X[X.columns[-1]]
        features = X.iloc[:, :-1]
        
        for group in groups.unique():
            group_mask = (groups == group)
            group_features = features[group_mask]
            
            # If the group has more samples than max_samples, take a random sample
            if len(group_features) > self.max_samples:
                group_features = group_features.sample(n=self.max_samples, random_state=self.random_state)
            
            iso_forest = IsolationForest(contamination=self.contamination, 
                                         random_state=self.random_state, 
                                         max_samples=min(len(group_features), 256),
                                         n_estimators=self.n_estimators, n_jobs=-1)
            iso_forest.fit(group_features)
            self.group_models[group] = iso_forest
        
        return self

    def transform(self, X):
        groups = X[X.columns[-1]]
        features = X.iloc[:, :-1]
        outlier_scores = np.zeros(X.shape[0])
        
        for group in groups.unique():
            group_mask = (groups == group)
            group_features = features[group_mask]
            
            if group in self.group_models:
                outlier_scores[group_mask] = self.group_models[group].decision_function(group_features)
            else:
                # If we encounter a new group during transform, we'll use the global mean
                outlier_scores[group_mask] = np.mean(list(self.group_models.values())[0].decision_function(group_features))
        
        return outlier_scores.reshape(-1, 1)
        
class GroupwiseLOF(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=20, contamination=0.1, min_group_size=10):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.min_group_size = min_group_size
        self.group_models = {}

    def fit(self, X, y=None):
        groups = X[X.columns[-1]]
        features = X.iloc[:, :-1]
        
        for group in groups.unique():
            group_mask = (groups == group)
            group_features = features[group_mask]
            
            if len(group_features) >= self.min_group_size:
                n_neighbors = min(self.n_neighbors, len(group_features) - 1)
                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=self.contamination, n_jobs=-1)
                lof.fit(group_features)
                self.group_models[group] = lof
            else:
                print(f"Warning: Group {group} has fewer than {self.min_group_size} samples. Skipping LOF for this group.")
        
        return self

    def transform(self, X):
        groups = X[X.columns[-1]]
        features = X.iloc[:, :-1]
        outlier_scores = np.zeros(X.shape[0])
        
        for group in groups.unique():
            group_mask = (groups == group)
            group_features = features[group_mask]
            
            if group in self.group_models:
                outlier_scores[group_mask] = -self.group_models[group].negative_outlier_factor_
            else:
                # For groups without a model (due to small size), assign a neutral score
                outlier_scores[group_mask] = 0
        
        return outlier_scores.reshape(-1, 1)

    def transform(self, X):
        groups = X[X.columns[-1]]
        features = X.iloc[:, :-1]
        outlier_scores = np.zeros(X.shape[0])
        
        for group in groups.unique():
            group_mask = (groups == group)
            group_features = features[group_mask]
            
            if group in self.group_models:
                outlier_scores[group_mask] = -self.group_models[group].negative_outlier_factor_
            else:
                # If we encounter a new group during transform, we'll use the global mean
                outlier_scores[group_mask] = np.mean(-list(self.group_models.values())[0].negative_outlier_factor_)
        
        return outlier_scores.reshape(-1, 1)


class TextOutlierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, z_score_threshold=1.5):
        self.z_score_threshold = z_score_threshold
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    def fit(self, X, y=None):
        combined_text = X['combined_text_review'].values
        self.vectorizer.fit(combined_text)
        return self

    def transform(self, X):
        combined_text = X['combined_text_review'].values
        description = X['combined_text_product'].values
        parent_asin = X['parent_asin'].values
        
        combined_text_tfidf = self.vectorizer.transform(combined_text)
        description_tfidf = self.vectorizer.transform(description)
        
        outlier_scores = []
        
        for group_id in np.unique(parent_asin):
            group_mask = (parent_asin == group_id)
            group_texts = combined_text_tfidf[group_mask]
            group_description = description_tfidf[group_mask][0]  # Assuming description is the same for all in group
            
            cosine_sim = cosine_similarity(group_texts, group_description)
            
            mean_sim = np.mean(cosine_sim)
            std_sim = np.std(cosine_sim)
            z_scores = (cosine_sim - mean_sim) / std_sim
            
            group_outlier_scores = (z_scores < -self.z_score_threshold).astype(float)
            outlier_scores.extend(group_outlier_scores)
        
        return np.array(outlier_scores).reshape(-1, 1)