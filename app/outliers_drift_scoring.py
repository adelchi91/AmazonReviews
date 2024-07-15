import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import ValidationError
import logging
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.helpers import (GroupwiseIsolationForest,
                            GroupwiseLOF, TextOutlierTransformer, 
                            TemporalOutlierDetector, BehavioralOutlierDetector, preprocess_column)
from src.pydantic_model import (ImagesMeta, Videos, InitialDataFrame, validate_dataframe, OutlierDetectedDataFrame,
                                OutlierDetectionRequest, DriftDetectionResult, DriftDetectionResponse, OutlierDetectionResponse, 
                                OutlierDetectionResult)
from src.outliers_drift_detector import (load_data, preprocess_data, detect_outliers, detect_drift)

# to keep track of df processing
processed_df = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# API endpoints
@app.post("/detect_outliers", response_model=OutlierDetectionResponse)
def detect_outliers_api():
    global processed_df
    try:
        # Hardcode the file path
        hardcoded_file_path = "data/amazon_reviews_beauty.joblib"
        
        # Load and preprocess the data
        df = load_data(hardcoded_file_path)
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

@app.post("/detect_drift", response_model=DriftDetectionResponse)
def detect_drift_api():
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
        test_df = detect_drift(train_df, test_df)
        
        # Extract the required columns
        drift_scores = test_df['drift_score'].tolist()
        has_drifted = test_df['has_drifted'].tolist()
        has_drifted_empirical = test_df['has_drifted_empirical'].tolist()
        
        return DriftDetectionResponse(
            status="success",
            drift_scores=drift_scores,
            has_drifted=has_drifted,
            has_drifted_empirical=has_drifted_empirical
        )
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in drift detection")


# Main function to run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    
    # Run FastAPI server using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)