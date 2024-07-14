import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import logging
from tqdm import tqdm
from alibi_detect.cd import TabularDrift
from ..src.helpers import (GroupwiseIsolationForest,
                            GroupwiseLOF, TextOutlierTransformer, 
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
    logger.info("Performing distribution drift analysis...")
    # Preprocess the data
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(preprocess_column)

    # Separate the data into training and test sets
    train_df = df[df['is_outlier'] == 0]
    test_df = df[df['is_outlier'] == 1]

    # Select features for drift analysis
    numerical_features = ['rating', 'helpful_vote', 'verified_purchase', 'price', 'average_rating', 'rating_number']
    categorical_features = ['main_category', 'store']
    text_features = ['combined_text_review', 'combined_text_product']

    # Ensure all selected features are present in both datasets
    numerical_features = [col for col in numerical_features if col in train_df.columns and col in test_df.columns]
    categorical_features = [col for col in categorical_features if col in train_df.columns and col in test_df.columns]
    text_features = [col for col in text_features if col in train_df.columns and col in test_df.columns]

    # Create column mapping
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features
    column_mapping.text_features = text_features

    # Use only numerical features for the drift detection
    train_data = train_df[numerical_features].values
    test_data = test_df[numerical_features].values

    # Initialize the drift detector
    cd = TabularDrift(train_data)

    # Function to calculate drift score for each row
    def calculate_drift_score(row, detector):
        row = row.reshape(1, -1)  # Reshape row for prediction
        preds = detector.predict(row, drift_type='batch', return_p_val=True, return_distance=True)
        return preds['data']['distance'][0]

    def calculte_has_drifred(row, detector):
        row = row.reshape(1, -1)  # Reshape row for prediction
        # Assuming detector is a TabularDrift instance
        preds = cd.predict(test_data, drift_type='batch', return_p_val=True)
        # Get p-values
        p_vals = preds['data']['p_val']
        # Define a threshold for significance
        threshold = 0.0005
        # Mark observations in the test set as having drifted if their p-value is below the threshold
        return np.any(p_vals < threshold).astype(int)

    # Calculate drift score for each row in the test dataset
    test_df['drift_score'] = test_df[numerical_features].apply(lambda row: calculate_drift_score(row.values, cd), axis=1)
    test_df['has_drifted'] = test_df[numerical_features].apply(lambda row: calculte_has_drifred(row.values, cd), axis=1)
    threshold = 0.5
    test_df['has_drifted_empirical'] = np.where(test_df['drift_score'] > threshold, 1, 0)

    logger.info("Distribution drift analysis complete.")

# Initialize FastAPI
app = FastAPI()

# API request models
class OutlierDetectionRequest(BaseModel):
    file_path: str

class DriftDetectionRequest(BaseModel):
    train_file_path: str
    test_file_path: str

# API endpoints
@app.post("/detect_outliers")
def detect_outliers_api(request: OutlierDetectionRequest):
    try:
        df = load_data(request.file_path)
        df = preprocess_data(df)
        df = detect_outliers(df)
        result = df[['outlier_score', 'is_outlier']].to_dict(orient='records')
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error in outlier detection: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in outlier detection")

@app.post("/detect_drift")
def detect_drift_api(request: DriftDetectionRequest):
    try:
        train_df = load_data(request.train_file_path)
        test_df = load_data(request.test_file_path)
        detect_drift(train_df, test_df)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in drift detection")

# Visualizations endpoint (optional)
@app.get("/visualizations")
def get_visualizations():
    # Implement logic to return visualizations as files or image data
    # Ensure appropriate handling of visualizations as per FastAPI documentation
    pass

# Main function to run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    
    # Run FastAPI server using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)