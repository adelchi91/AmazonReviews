from pydantic import BaseModel, ValidationError, Field
from typing import List, Optional, Dict, Any
import json
import pandas as pd
import logging 
from datetime import datetime


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

#class DriftDetectionRequest(BaseModel):
#    train_file_path: str
#    test_file_path: str

class DriftDetectionResult(BaseModel):
    drift_score: float
    has_drifted: int
    has_drifted_empirical: int

class DriftDetectionResponse(BaseModel):
    status: str
    result: List[DriftDetectionResult]

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
    drift_scores: List[float]
    has_drifted: List[int]
    has_drifted_empirical: List[int]