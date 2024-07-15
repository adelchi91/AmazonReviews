# 1 Outliers detection methodology

We decided to tackle the outliers detection by taking three types of approaches:
1. A user experienced/business based approach
2. A statistical/ML approach
3. A NLP approach comparing product description and customer reviews text fields 


## 1.1 User experienced approach

### Behavioral Outlier Detector:

We've added a `BehavioralOutlierDetector` class that identifies two types of behavioral outliers:

1. Users who post an unusually high number of reviews in a short time (high_frequency_outlier).

2. Users who consistently give ratings that deviate significantly from the average (rating_deviation_outlier).


### High Frequency Outlier Detection:

It groups reviews by user and time window (default is daily).
Users who post more than a threshold number of reviews (default is 5) in a single time window are flagged as high_frequency_outliers.


### Rating Deviation Outlier Detection:

It calculates the average rating for each user and compares it to the overall average rating.
Users whose average rating deviates from the overall average by more than a threshold (default is 1.5) are flagged as rating_deviation_outliers.


### Temporal Outlier Detection:

We've added a TemporalOutlierDetector class that identifies temporal outliers.
It groups reviews by a specified time window (default is daily) and flags reviews that occur on days with an unusually high number of reviews (above the 95th percentile).


## 1.2 Statistical/ML approach

A classical methodology would leverage the following:

- Z-Score Method (for numerical features)
- Isolation Forest (for multi-dimensional outlier detection)
- Local Outlier Factor (LOF) (for density-based outlier detection)

However we decided to drop the Z-score method as it assumes a normal distribution of the data.

### Isolation Forest:

This is an unsupervised learning algorithm that isolates anomalies in the dataset.
It works well with high-dimensional datasets and doesn't make assumptions about the distribution of the data.
We apply it to a selection of relevant features, including both numerical and categorical (encoded) data.


### Local Outlier Factor (LOF):

LOF is a density-based method that compares the local density of a point to the local densities of its neighbors.
It's effective at finding outliers in datasets with varying densities.
Like Isolation Forest, we apply it to a selection of relevant features.

## 1.3 NLP approach 

We created a class for text outliers in review data based on cosine similarity.

The class identifies text outliers by comparing the TF-IDF vectors of review texts with their corresponding product descriptions, using cosine similarity and z-scores.

We combine all relevant text fields (title, review text, main category, and product title) for product on one hand and for the custumer review on the other hand, into a single text field each. 

We use TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize the text data. This method captures the importance of words in the context of the entire corpus.
We compute the cosine similarity between each review and all other reviews.

To further improve this analysis, we could:

* Adjust the thresholds and parameters based on domain knowledge and exploratory data analysis.
* Experiment with different text vectorization methods (e.g., word embeddings like Word2Vec or BERT).
Use topic modeling techniques (e.g., LDA) to identify reviews that don't fit well into any common topics.
* Conduct a manual review of the top outliers to understand why they were flagged and to validate the approach.


## 1.4 Metrics and Evaluation:

We're using the proportion of samples identified as outliers by each method as a basic metric.
We also create an 'outlier_score' by summing the results of all three methods, allowing us to rank the samples most likely to be outliers.

This approach provides a comprehensive view of potential outliers in the dataset. The combination of methods helps to catch different types of outliers:

Isolation Forest and LOF catch multivariate outliers and can work with both numerical and categorical data.

To further improve this analysis we could:

* Adjust the thresholds and parameters based on domain knowledge and exploratory data analysis.
* Investigate the top outliers manually to understand why they were flagged.
* Consider the context of the data (e.g., some "outliers" might be legitimate extreme reviews rather than errors).


# 2. Running api with Docker

If you have not started our docker, run in the terminal as root

`sudo systemctl start docker`

build your image in the root folder 

`docker build -t amazon-reviews-app .`

and finally run the api

`docker run -p 8000:8000 amazon-reviews-app`

add `/docs` to the URL for the swagger, i.e 

`http://0.0.0.0:8000/docs#`
