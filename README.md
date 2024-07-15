# Methodology
For outlier detection in a dataset like Amazon reviews, we can use a combination of statistical and machine learning methods. Here's a suggested strategy along with code implementation:

- Z-Score Method (for numerical features)
- Isolation Forest (for multi-dimensional outlier detection)
- Local Outlier Factor (LOF) (for density-based outlier detection)


## Z-Score Method:

This method assumes a normal distribution of the data.
It calculates how many standard deviations a data point is from the mean.
We consider a point an outlier if its z-score is above a threshold (typically 3).
This method is applied to numerical features individually.


## Isolation Forest:

This is an unsupervised learning algorithm that isolates anomalies in the dataset.
It works well with high-dimensional datasets and doesn't make assumptions about the distribution of the data.
We apply it to a selection of relevant features, including both numerical and categorical (encoded) data.


## Local Outlier Factor (LOF):

LOF is a density-based method that compares the local density of a point to the local densities of its neighbors.
It's effective at finding outliers in datasets with varying densities.
Like Isolation Forest, we apply it to a selection of relevant features.



## Metrics and Evaluation:

We're using the proportion of samples identified as outliers by each method as a basic metric.
We also create an 'outlier_score' by summing the results of all three methods, allowing us to rank the samples most likely to be outliers.

This approach provides a comprehensive view of potential outliers in the dataset. The combination of methods helps to catch different types of outliers:

Z-Score catches univariate outliers in numerical features.
Isolation Forest and LOF catch multivariate outliers and can work with both numerical and categorical data.

To further improve this analysis, you could:

Adjust the thresholds and parameters based on domain knowledge and exploratory data analysis.
Investigate the top outliers manually to understand why they were flagged.
Consider the context of the data (e.g., some "outliers" might be legitimate extreme reviews rather than errors).
Use cross-validation to ensure the stability of your outlier detection results.


## Text-based outlier detection using TF-IDF and cosine similarity

We combine all relevant text fields (title, review text, main category, and product title) into a single text field.
We use TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize the text data. This method captures the importance of words in the context of the entire corpus.
We compute the cosine similarity between each review and all other reviews.
We calculate the average similarity for each review.
Reviews with an average similarity below a certain threshold are considered outliers.

To further improve this analysis, you could:

Adjust the thresholds and parameters based on domain knowledge and exploratory data analysis.
Experiment with different text vectorization methods (e.g., word embeddings like Word2Vec or BERT).
Use topic modeling techniques (e.g., LDA) to identify reviews that don't fit well into any common topics.
Implement a weighted scoring system if you believe some methods are more reliable than others.
Conduct a manual review of the top outliers to understand why they were flagged and to validate the approach.


# Other notes

Given the available features in both the user reviews and item metadata datasets, we can consider several factors to identify outliers. Here's a breakdown of potential outlier indicators:

Numerical Outliers:

Rating: Reviews with ratings that significantly deviate from the product's average rating.
Helpful Votes: Reviews with an unusually high or low number of helpful votes.
Price: Products with prices that are significantly higher or lower than others in the same category.
Rating Number: Products with an unusually high or low number of ratings compared to similar products.


Textual Outliers:

Review Length: Extremely short or long reviews compared to the average.
Review Content: Reviews that use language or discuss topics that are significantly different from other reviews for the same or similar products.
Product Description: Products with unusually short or long descriptions.


Categorical Outliers:

Verified Purchase: If a disproportionate number of reviews for a product are unverified.
Main Category: Products that seem misclassified based on their features or description.


Temporal Outliers:

Review Timestamp: Reviews posted at unusual times or in unusual patterns.


Relational Outliers:

Mismatch between review content and product features or description.
Inconsistency between review rating and review text sentiment.


Image-related Outliers:

Products or reviews with an unusually high or low number of images.
Reviews with images that are inconsistent with the product category.


Behavioral Outliers:

Users who post an unusually high number of reviews in a short time.
Users who consistently give ratings that deviate significantly from the average.


Product Characteristic Outliers:

Products with an unusually high or low number of features.
Products with a significantly different price compared to others in the same category.


Recommendation Outliers:

Products with "bought together" recommendations that seem unrelated or unusual for the category.

**should treat the categoriesof Numerical Outliers, Textual outliers, RElational outlies**

Explanation of changes and additions:

Temporal Outlier Detection:

We've added a TemporalOutlierDetector class that identifies temporal outliers.
It groups reviews by a specified time window (default is daily) and flags reviews that occur on days with an unusually high number of reviews (above the 95th percentile).


Scikit-learn Pipeline:

We've created a pipeline that combines all our outlier detection methods.
The ColumnTransformer is used to apply different transformations to different types of data (numerical, text, temporal).
The pipeline includes the Isolation Forest and Local Outlier Factor methods.


Text Outlier Detection:

The text outlier detection function is now integrated into the pipeline using a FunctionTransformer.


Data Preparation:

We've moved the data preparation steps (converting to numeric, filling NAs, creating combined text) before applying the pipeline.

Behavioral Outliers: 
* Users who post an unusually high number of reviews in a short time.
* Users who consistently give ratings that deviate significantly from the average.

Behavioral Outlier Detector:

We've added a BehavioralOutlierDetector class that identifies two types of behavioral outliers:
a. Users who post an unusually high number of reviews in a short time (high_frequency_outlier).
b. Users who consistently give ratings that deviate significantly from the average (rating_deviation_outlier).


High Frequency Outlier Detection:

It groups reviews by user and time window (default is daily).
Users who post more than a threshold number of reviews (default is 5) in a single time window are flagged as high_frequency_outliers.


Rating Deviation Outlier Detection:

It calculates the average rating for each user and compares it to the overall average rating.
Users whose average rating deviates from the overall average by more than a threshold (default is 1.5) are flagged as rating_deviation_outliers.

# Launch Dockerfile

`sudo systemctl start docker`

`docker build -t amazon-reviews-app .`

`docker run -p 8000:8000 amazon-reviews-app`
