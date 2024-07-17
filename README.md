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

We created a final voting mechanism, by combining all outlier methods scores into one, where we consider the review to be an outlier if flagged by at least two methods. The method should be then more robust. 

This approach provides a comprehensive view of potential outliers in the dataset. The combination of methods helps to catch different types of outliers:

Isolation Forest and LOF catch multivariate outliers and can work with both numerical and categorical data.

To further improve this analysis we could:

* Adjust the thresholds and parameters based on domain knowledge and exploratory data analysis.
* Investigate the top outliers manually to understand why they were flagged.
* Consider the context of the data (e.g., some "outliers" might be legitimate extreme reviews rather than errors).


# 2. Creating a virtual enironment

To ensure consistency and isolate our project dependencies, we'll use a virtual environment. Follow these steps to set it up:


## 2.1 Create a Virtual Environment
Open your terminal and navigate to your project directory. Then run:

```bash
python -m venv venv
```
## 2.2 Activate the Virtual Environment

Activate the virtual environment:

On Windows:
```bash
venv\Scripts\activate
```

On macOS and Linux:
```bash
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your terminal prompt, indicating that the virtual environment is active.

## 2.3 Install Dependencies

With the virtual environment activated, install the dependencies:

```bash
pip install -r requirements.txt
```

# 3. Running api with Docker

If you have not started our docker, run in the terminal as root

```
sudo systemctl start docker
```

build your image in the root folder 

```
docker build -t amazon-reviews-app .
```

and finally run the api

```
docker run -p 8000:8000 amazon-reviews-app
```

add `/docs` to the URL for the swagger, i.e 

```
http://0.0.0.0:8000/docs#
```

# 4. Folder structure 

Our project follows a structured layout to organize code, data, and documentation. Here's an overview of the directory structure:

- `app/`: Contains the main application script for outlier and drift scoring. It is implemented with `FastAPI`. It contains the data validation (carried out with `Pydantic`) and exposes the api with different endpoints. 
- `data/`: Stores the dataset and analysis results.
- `figures/`: Holds visualizations for drift and outlier analysis.
- `Instructions/`: Contains project instructions.
- `notebooks/`: Jupyter notebooks for data exploration and analysis. The final version cleaned is `Exploration_clean.ipynb`, which is fully commented with markdown cells. 
- `src/`: Source code for helper functions, detectors, and data models. It contains the code leveraged both in the app and notebooks. 
- `Dockerfile`: Defines the Docker container for the project.
- `README.md`: Project overview and setup instructions.
- `requirements.txt`: List of Python package dependencies.

This structure separates concerns, keeping data, code, and documentation organized. The `src/` directory contains reusable modules, while `notebooks/` is used for exploratory and visualiation work. The `app/` directory holds the main application script, making it easy to identify the entry point of the project.



# 5. Disclaimer and project improvements

I am currently running a Fedora laptop with the configuration presented below. As such I have made a deliberate choice to sample 10000 reviews in order for me to work with the resources available to me. This can lead to wrong results, considering that outliers are scarce. Other possible sampling approaches could have been 

* Bootstrapping: Use bootstrapping techniques to create multiple samples and assess the stability of your results. This can help us understand the variability in our estimates.

* Time-based sampling: If the dataset spans a long period, ensure we sample reviews from different time periods to capture potential temporal variations.

## Technical specs: ##

```bash
cat /etc/os-release
```
### OS Information ###
```bash
NAME="Fedora Linux"
VERSION="38 (Workstation Edition)"
ID=fedora
VERSION_ID=38
VERSION_CODENAME=""
PLATFORM_ID="platform:f38"
PRETTY_NAME="Fedora Linux 38 (Workstation Edition)"
ANSI_COLOR="0;38;2;60;110;180"
LOGO=fedora-logo-icon
CPE_NAME="cpe:/o:fedoraproject:fedora:38"
DEFAULT_HOSTNAME="fedora"
HOME_URL="https://fedoraproject.org/"
DOCUMENTATION_URL="https://docs.fedoraproject.org/en-US/fedora/f38/system-administrators-guide/"
SUPPORT_URL="https://ask.fedoraproject.org/"
BUG_REPORT_URL="https://bugzilla.redhat.com/"
REDHAT_BUGZILLA_PRODUCT="Fedora"
REDHAT_BUGZILLA_PRODUCT_VERSION=38
REDHAT_SUPPORT_PRODUCT="Fedora"
REDHAT_SUPPORT_PRODUCT_VERSION=38
SUPPORT_END=2024-05-21
VARIANT="Workstation Edition"
VARIANT_ID=workstation
```

```bash
echo -e "\n### RAM Information ###"
free -h
```
### RAM Information ###
```
               total        used        free      shared    buff/cache   available
Mem:            11Gi       4.7Gi       456Mi        620Mi       6.5Gi       6.0Gi
Swap:          8.0Gi          0B       8.0Gi
```
```
echo "### CPU Information ###"
lscpu
```

### CPU Information ###
```bash
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          39 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   4
  On-line CPU(s) list:    0-3
Vendor ID:                GenuineIntel
  Model name:             Intel(R) Core(TM) i5-4310M CPU @ 2.70GHz
    CPU family:           6
    Model:                60
    Thread(s) per core:   2
    Core(s) per socket:   2
    Socket(s):            1
    Stepping:             3
    CPU(s) scaling MHz:   95%
    CPU max MHz:          3400.0000
    CPU min MHz:          800.0000
    BogoMIPS:             5387.92
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss
                           ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_ts
                          c cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse
                          4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm cpuid_fault epb pti s
                          sbd ibrs ibpb stibp tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invp
                          cid xsaveopt dtherm ida arat pln pts vnmi md_clear flush_l1d
Virtualization features:  
  Virtualization:         VT-x
Caches (sum of all):      
  L1d:                    64 KiB (2 instances)
  L1i:                    64 KiB (2 instances)
  L2:                     512 KiB (2 instances)
  L3:                     3 MiB (1 instance)
NUMA:                     
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-3
Vulnerabilities:          
  Gather data sampling:   Not affected
  Itlb multihit:          KVM: Mitigation: VMX disabled
  L1tf:                   Mitigation; PTE Inversion; VMX conditional cache flushes, SMT vulnerable
  Mds:                    Mitigation; Clear CPU buffers; SMT vulnerable
  Meltdown:               Mitigation; PTI
  Mmio stale data:        Unknown: No mitigations
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Not affected
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:             Mitigation; Retpolines; IBPB conditional; IBRS_FW; STIBP conditional; RSB filling; PBRSB-eIBRS Not affected; 
                          BHI Not affected
  Srbds:                  Mitigation; Microcode
  Tsx async abort:        Not affected
```

The sampling was carried out using 

```python
full_path = "../data/amazon_reviews_beauty.joblib"
file_name = "amazon_reviews_beauty.joblib"
if os.path.exists(full_path):
    print(f"The file {file_name} exists in the folder.")
    # Load the joblib file
    df = joblib.load('../data/amazon_reviews_beauty.joblib')
else:
    print(f"The file {file_name} does not exists in the folder. Importing...")
    # Load the review dataset
    review_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", split="full", trust_remote_code=True)
    # Load the metadata dataset
    meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True)
    # Convert datasets to pandas DataFrames
    df_reviews = pd.DataFrame(review_dataset).sample(n=10000, random_state=2024)
    df_meta = pd.DataFrame(meta_dataset)
    # Merge the datasets on parent_asin
    df = pd.merge(df_reviews, df_meta, on='parent_asin', how='left', suffixes=('_review', '_meta'))
    # Save the DataFrame as a joblib file
    joblib.dump(df, '../data/amazon_reviews_beauty.joblib') 
```

To enhance the project's quality, maintainability, and deployment process, we propose the following improvements:

## 5.1. GitHub Actions Pipeline

Implement a GitHub Actions workflow to automate various tasks:

### Code Quality Checks
- **isort**: Automatically sorts import statements in Python files.
  - Ensures consistent import ordering across the project.
- **flake8**: Lints Python code for style and potential errors.
  - Enforces PEP 8 style guide and catches common programming errors.
- **black**: An opinionated code formatter for Python.
  - Automatically formats code to a consistent style, reducing debates about formatting.

### Docker Image Handling
- Build the Docker image
- Push the image to DockerHub
  - Makes the image available for easy deployment to cloud platforms like AWS or Azure.

### Documentation Generation
Automatically generate and update project documentation.

We could possiby use either Sphinx or MkDocs:

#### Sphinx

* A powerful documentation generator that uses reStructuredText as its markup language.
Features:

* Outputs in various formats (HTML, PDF, ePub)
    * Extensive cross-referencing features
    * Hierarchical structure of the documentation
    * Code documentation extraction from Python docstrings



#### MkDocs

* A fast, simple static site generator geared towards building project documentation.
* Features:

    * Uses Markdown for easier writing
    * Live preview of documentation changes
    * Easy-to-customize themes
    * Built-in dev-server for instant previews

Both tools can:

Generate API documentation from code comments
Create a searchable documentation website
Be easily integrated with Read the Docs for hosting

## 5.2 Example GitHub Actions Workflow

```yaml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.x'
      - name: Install dependencies
        run: |
          pip install isort flake8 black
      - name: Run isort
        run: isort .
      - name: Run flake8
        run: flake8 .
      - name: Run black
        run: black . --check

  build-and-push:
    needs: code-quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t myproject:latest .
      - name: Push to DockerHub
        run: |
          echo ${{ secrets.DOCKERHUB_TOKEN }} | docker login -u ${{ secrets.DOCKERHUB_USERNAME }} --password-stdin
          docker push myproject:latest

  generate-docs:
    needs: code-quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.x'
      - name: Install documentation tools
        run: pip install sphinx  # or mkdocs
      - name: Generate documentation
        run: sphinx-build -b html docs/ docs/_build  # adjust as needed
```

