import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ValidationError, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from tqdm import tqdm
from alibi_detect.cd import TabularDrift
import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.helpers import (GroupwiseIsolationForest,
                            GroupwiseLOF, TextOutlierTransformer, 
                            TemporalOutlierDetector, BehavioralOutlierDetector, preprocess_column)

# to keep track of df processing
processed_df = None

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
    # Select features for drift analysis
    numerical_features = ['rating', 'helpful_vote', 'verified_purchase', 'price', 'average_rating', 'rating_number']
    categorical_features = ['main_category', 'store']
    text_features = ['combined_text_review', 'combined_text_product']

    # Ensure all selected features are present in both datasets
    numerical_features = [col for col in numerical_features if col in train_df.columns and col in test_df.columns]
    categorical_features = [col for col in categorical_features if col in train_df.columns and col in test_df.columns]
    text_features = [col for col in text_features if col in train_df.columns and col in test_df.columns]

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
        threshold = 0.05
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

class ImagesMeta(BaseModel):
    hi_res: List[Optional[str]]
    large: List[Optional[str]]
    thumb: List[Optional[str]]
    variant: List[str]

class Videos(BaseModel):
    title: List
    url: List
    user_id: List

class InitialDataFrame(BaseModel):
    rating: float
    title_review: str
    text: str
    images_review: List = []
    asin: str
    parent_asin: str
    user_id: str
    timestamp: int
    helpful_vote: int
    verified_purchase: bool
    main_category: str
    title_meta: str
    average_rating: float
    rating_number: int
    features: List = []
    description: List = []
    price: Optional[str]
    images_meta: ImagesMeta
    videos: Videos
    store: Optional[str]
    categories: List = []
    details: Optional[Dict[str, Any]]  # Change this to Dict
    bought_together: Optional[str]
    subtitle: Optional[str]
    author: Optional[str]

    class Config:
        extra = "forbid"

    @classmethod
    def parse_obj(cls, obj):
        if 'details' in obj and isinstance(obj['details'], str):
            obj['details'] = json.loads(obj['details'])
        return super().parse_obj(obj)
    

# to validate with pydantic model
def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        data_dict = df.to_dict(orient='records')
        validated_data = []
        for row in data_dict:
            # Handle None values in 'store' field
            if row['store'] is None:
                row['store'] = "Unknown"
            # Convert 'None' string to None
            if row['price'] == 'None':
                row['price'] = None
            
            # Parse the details JSON string
            if row['details']:
                try:
                    row['details'] = json.loads(row['details'])
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in details field: {row['details']}")
                    row['details'] = None
            
            # Handle potential None values in images_meta
            if 'images_meta' in row:
                for key in ['hi_res', 'large', 'thumb']:
                    if key in row['images_meta']:
                        row['images_meta'][key] = [url if url is not None else '' for url in row['images_meta'][key]]
            
            # Validate the row
            validated_row = InitialDataFrame.parse_obj(row).dict()
            validated_data.append(validated_row)
        
        # Convert back to DataFrame
        return pd.DataFrame(validated_data)
    except ValidationError as e:
        logger.error(f"Data validation error: {e}")
        raise


# Model for the dataframe after outlier detection
class OutlierDetectedDataFrame(InitialDataFrame):
    combined_text_review: str
    combined_text_product: str
    isolation_forest_outlier: int
    lof_outlier: int
    text_outlier: int
    temporal_outlier: int
    high_frequency_outlier: int
    rating_deviation_outlier: int
    outlier_score: int
    is_outlier: int
    timestamp: datetime  # Note: This field is changed to datetime
    verified_purchase: int  # Note: This field is changed to int
    price: float  # Note: This field is changed to float

# API request models
class OutlierDetectionRequest(BaseModel):
    file_path: str

class DriftDetectionRequest(BaseModel):
    train_file_path: str
    test_file_path: str

# Model for a single row of the outlier detection result
class OutlierDetectionResult(BaseModel):
    outlier_score: int
    is_outlier: int

# Model for the response of the outlier detection API
class OutlierDetectionResponse(BaseModel):
    status: str
    result: List[OutlierDetectionResult]

# Model for the response of the drift detection API
class DriftDetectionResponse(BaseModel):
    status: str

# API endpoints
@app.post("/detect_outliers", response_model=OutlierDetectionResponse)
def detect_outliers_api():
    global processed_df
    try:
        # Hardcode the file path
        hardcoded_file_path = "data/amazon_reviews_beauty.joblib"
        
        # Load and preprocess the data
        df = load_data(hardcoded_file_path)
        print('++++++++++++++++++++++++++++++++++++++++')
        print(df['details'].head())
        print('++++++++++++++++++++++++++++++++++++++++')
        # Validate the raw data using the Pydantic model
        validated_data = validate_dataframe(df)
        # Preprocess the validated data
        df = preprocess_data(validated_data)
        # Perform outlier detection
        df = detect_outliers(df)
        processed_df = df  # Store the processed dataframe
        
        result = df[['outlier_score', 'is_outlier']].to_dict(orient='records')
        return OutlierDetectionResponse(status="success", result=result)
    except ValidationError as e:
        logger.error(f"Data validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in outlier detection: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in outlier detection")

@app.post("/detect_drift")
def detect_drift_api(request: DriftDetectionRequest):
    global processed_df
    try:
        if processed_df is None:
            raise HTTPException(status_code=400, detail="Please run outlier detection first")

        df = processed_df.copy()  # Use the stored dataframe
        
        # Preprocess the data
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(preprocess_column)

        # Separate the data into training and test sets
        train_df = df[df['is_outlier'] == 0]
        test_df = df[df['is_outlier'] == 1]
        detect_drift(train_df, test_df)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in drift detection")


# Main function to run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    
    # Run FastAPI server using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)