import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import joblib
import pandas as pd
import logging
import os
import sys
from tqdm import tqdm
from alibi_detect.cd import TabularDrift

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.helpers import (TextOutlierTransformer, 
                        TemporalOutlierDetector, BehavioralOutlierDetector, preprocess_column)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data function
def load_data(file_path):
    if os.path.exists(file_path):
        logger.info(f"Loading data from {file_path}...")
        df = joblib.load(file_path)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return df

# Preprocess data function
def preprocess_data(df):
    logger.info("Preprocessing data...")
    df['helpful_vote'] = pd.to_numeric(df['helpful_vote'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['verified_purchase'] = df['verified_purchase'].astype(int)
    
    # Ensure text columns are converted to strings
    text_columns = ['title_review', 'text', 'main_category', 'title_meta', 'description']
    for col in text_columns:
        df[col] = df[col].astype(str)
    df['combined_text_review'] = df['title_review'] + ' ' + df['text']
    df['combined_text_product'] = df['main_category'] + ' ' + df['title_meta'] + ' ' + df['description']
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())
    
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_columns:
        df[col] = df[col].fillna("Unknown")
    
    logger.info("Data preprocessing complete.")
    return df

# Outlier detection function
def detect_outliers(df):
    logger.info("Running outlier detection...")

    # Feature pipeline
    feature_pipeline = ColumnTransformer([
    ('num', StandardScaler(), ['rating', 'helpful_vote', 'verified_purchase', 'price', 'average_rating', 'rating_number']),
    ('text', TextOutlierTransformer(z_score_threshold=1.5), ['combined_text_review', 'combined_text_product', 'parent_asin']),
    ('temporal', TemporalOutlierDetector(group_column='parent_asin'), ['timestamp', 'parent_asin']),
    ('behavioral', BehavioralOutlierDetector(group_column='parent_asin'), ['user_id', 'timestamp', 'rating', 'parent_asin']),
    ])

    with tqdm(total=5, desc="Processing") as pbar:
        X_transformed = feature_pipeline.fit_transform(df)
        pbar.update(1)

        X_transformed = np.asarray(X_transformed)
        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(-1, 1)

        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        iso_forest_outliers = iso_forest.fit_predict(X_transformed)
        pbar.update(1)

        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, n_jobs=-1)
        lof_outliers = lof.fit_predict(X_transformed)
        pbar.update(1)

        df['isolation_forest_outlier'] = (iso_forest_outliers == -1).astype(int)
        df['lof_outlier'] = (lof_outliers == -1).astype(int)
        df['text_outlier'] = (X_transformed[:, 0] > 0.5).astype(int)
        df['temporal_outlier'] = (X_transformed[:, 1] > 0.5).astype(int)
        df['high_frequency_outlier'] = (X_transformed[:, 2] > 0.5).astype(int)
        df['rating_deviation_outlier'] = (X_transformed[:, 3] > 0.5).astype(int)
        pbar.update(1)

        df['outlier_score'] = (df['isolation_forest_outlier'] + 
                            df['lof_outlier'] + 
                            df['text_outlier'] + 
                            df['temporal_outlier'] + 
                            df['high_frequency_outlier'] + 
                            df['rating_deviation_outlier'])
        df['is_outlier'] = (df['outlier_score'] >= 2).astype(int)
        pbar.update(1)
    
    logger.info("Outlier detection complete.")
    return df

# Drift detection function
def detect_drift(train_df, test_df):
    # logger.info("Performing distribution drift analysis...")
    
    # Select features for drift analysis
    numerical_features = ['rating', 'helpful_vote', 'verified_purchase', 'price', 'average_rating', 'rating_number']
    categorical_features = ['main_category', 'store']
    text_features = ['combined_text_review', 'combined_text_product']
    
    # Ensure all selected features are present in both datasets
    numerical_features = [col for col in numerical_features if col in train_df.columns and col in test_df.columns]
    categorical_features = [col for col in categorical_features if col in train_df.columns and col in test_df.columns]
    text_features = [col for col in text_features if col in train_df.columns and col in test_df.columns]
    
    # # TF-IDF Vectorization for text features
    # tfidf_vectorizers = {}
    # for text_feature in text_features:
    #     vectorizer = TfidfVectorizer(max_features=100)  # You can adjust max_features as needed
    #     train_text_vector = vectorizer.fit_transform(train_df[text_feature].fillna(''))
    #     test_text_vector = vectorizer.transform(test_df[text_feature].fillna(''))
        
    #     # Add vectorized features to dataframes
    #     feature_names = [f"{text_feature}_{i}" for i in range(train_text_vector.shape[1])]
    #     train_df = train_df.join(pd.DataFrame(train_text_vector.toarray(), columns=feature_names, index=train_df.index))
    #     test_df = test_df.join(pd.DataFrame(test_text_vector.toarray(), columns=feature_names, index=test_df.index))
        
    #     # Update numerical_features list
    #     numerical_features.extend(feature_names)
        
    #     # Store vectorizer for later use if needed
    #     tfidf_vectorizers[text_feature] = vectorizer
    
    # Combine all features
    all_features = numerical_features # + categorical_features
    
    # Prepare the data
    train_data = train_df[all_features]
    test_data = test_df[all_features]
    
    # Create categories_per_feature dictionary
    categories_per_feature = {}
    for i, feature in enumerate(all_features):
        if feature in categorical_features:
            categories_per_feature[i] = None
    
    # Initialize the drift detector - KSDrift` will be applied to all features.
    cd = TabularDrift(
        train_data.to_numpy(),
        p_val=.05,
        #categories_per_feature=categories_per_feature
    )
    
    # Function to calculate drift score for each row
    def calculate_drift_score(row, detector):
        row = row.values.reshape(1, -1)  # Reshape row for prediction
        preds = detector.predict(row, return_p_val=True, return_distance=True)
        return preds['data']['distance'].mean()  # Average distance across features
    
    # Calculate drift score for each row in both datasets
    test_df['drift_score'] = test_data.apply(lambda row: calculate_drift_score(row, cd), axis=1)
    train_df['drift_score'] = train_data.apply(lambda row: calculate_drift_score(row, cd), axis=1)
    
    # Calculate has_drifted for the entire test set
    preds = cd.predict(test_data.to_numpy(), return_p_val=True)
    test_df['has_drifted'] = int(preds['data']['is_drift'])
    
    # Calculate empirical drift
    threshold = 0.5
    test_df['has_drifted_empirical'] = np.where(test_df['drift_score'] > threshold, 1, 0)
    
    # logger.info("Distribution drift analysis complete.")
    return test_df, train_df