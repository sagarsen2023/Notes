# 1. Introduction to Data Warehousing & Data Mining

## 1.1 Data Warehousing Concepts

A data warehouse is a centralized repository that stores integrated data from multiple sources for reporting and analysis purposes. It is designed to support business intelligence activities, particularly analytical processing.

### Key Characteristics of Data Warehouses:
- **Subject-oriented**: Organized around major subjects like customers, products, or sales
- **Integrated**: Consolidates data from various sources into a consistent format
- **Time-variant**: Contains historical data for trend analysis
- **Non-volatile**: Data is stable and doesn't change once loaded
- **Supports decision-making**: Optimized for query and analysis rather than transaction processing

## 1.2 Data Mining Overview

Data mining is the process of discovering patterns, correlations, and insights from large datasets. It combines techniques from statistics, machine learning, and database systems to extract knowledge from data.

### Data Mining Tasks:
- **Descriptive**: Summarizing and visualizing data characteristics
- **Predictive**: Forecasting future trends and behaviors
- **Prescriptive**: Recommending actions based on predictions

### Data Mining Applications:
- Market basket analysis
- Customer segmentation
- Fraud detection
- Medical diagnosis
- Scientific research
- Social network analysis

## 1.3 Mining Frequent Patterns, Associations and Correlations

Frequent pattern mining discovers recurring relationships in datasets. These patterns reveal items or events that frequently co-occur.

### Association Rule Mining:
- **Definition**: Identifies relationships between items in large datasets (if X, then Y)
- **Key Metrics**:
- **Support**: Frequency of an itemset in the dataset
   - Example: If {bread, milk} appears in 20 out of 200 transactions, support = 20/200 = 10%
- **Confidence**: Likelihood that Y appears when X appears
   - Example: If {bread, milk} appears in 20 transactions and {bread} in 40 transactions, confidence of {bread} → {milk} = 20/40 = 50%
- **Lift**: Ratio of observed support to expected support if X and Y were independent
   - Example: If support({bread, milk}) = 10%, support({bread}) = 20%, and support({milk}) = 30%, lift = 10%/(20% × 30%) = 1.67

### Apriori Algorithm:
1. Generate frequent itemsets (items exceeding minimum support threshold)
2. Generate association rules from frequent itemsets
3. Prune rules below confidence threshold

#### Example of Apriori Algorithm:

Consider a small transaction database:
| Transaction ID | Items |
|----------------|-------|
| T1             | Bread, Milk, Butter |
| T2             | Bread, Diaper, Beer, Eggs |
| T3             | Milk, Diaper, Beer, Cola |
| T4             | Bread, Milk, Diaper, Beer |
| T5             | Bread, Milk, Diaper, Cola |

Let's apply Apriori with minimum support = 60% (3 transactions) and confidence = 70%:

**Step 1: Find 1-itemsets meeting minimum support**
- {Bread}: 4/5 = 80% ✓
- {Milk}: 4/5 = 80% ✓
- {Diaper}: 4/5 = 80% ✓
- {Beer}: 3/5 = 60% ✓
- {Butter}: 1/5 = 20% ✗
- {Eggs}: 1/5 = 20% ✗
- {Cola}: 2/5 = 40% ✗

**Step 2: Generate 2-itemsets from frequent 1-itemsets**
- {Bread, Milk}: 3/5 = 60% ✓
- {Bread, Diaper}: 3/5 = 60% ✓
- {Bread, Beer}: 2/5 = 40% ✗
- {Milk, Diaper}: 3/5 = 60% ✓
- {Milk, Beer}: 2/5 = 40% ✗
- {Diaper, Beer}: 3/5 = 60% ✓

**Step 3: Generate 3-itemsets from frequent 2-itemsets**
- {Bread, Milk, Diaper}: 2/5 = 40% ✗

**Step 4: Generate rules from frequent itemsets and check confidence**

From {Bread, Milk}:
- {Bread} → {Milk}: 3/4 = 75% ✓
- {Milk} → {Bread}: 3/4 = 75% ✓

From {Bread, Diaper}:
- {Bread} → {Diaper}: 3/4 = 75% ✓
- {Diaper} → {Bread}: 3/4 = 75% ✓

From {Milk, Diaper}:
- {Milk} → {Diaper}: 3/4 = 75% ✓
- {Diaper} → {Milk}: 3/4 = 75% ✓

From {Diaper, Beer}:
- {Diaper} → {Beer}: 3/4 = 75% ✓
- {Beer} → {Diaper}: 3/3 = 100% ✓

**Final association rules** (with support, confidence):
- {Bread} → {Milk}: (60%, 75%)
- {Milk} → {Bread}: (60%, 75%)
- {Bread} → {Diaper}: (60%, 75%)
- {Diaper} → {Bread}: (60%, 75%)
- {Milk} → {Diaper}: (60%, 75%)
- {Diaper} → {Milk}: (60%, 75%)
- {Diaper} → {Beer}: (60%, 75%)
- {Beer} → {Diaper}: (60%, 100%)

### FP-Growth Algorithm:
- More efficient than Apriori for large datasets
- Uses a compressed tree structure (FP-tree) to store dataset
- Eliminates need for multiple database scans

### Market Basket Analysis Example:
If customers buy bread and butter, they are 75% likely to also buy milk.
- Itemset: {bread, butter, milk}
- Rule: {bread, butter} → {milk}
- Support: 10% (percentage of transactions containing all three items)
- Confidence: 75% (percentage of transactions containing bread and butter that also contain milk)

## 1.4 Sequential Pattern Mining

Sequential pattern mining discovers frequent subsequences as patterns in a sequence database.
### Key Concepts:
- **Sequence**: Ordered list of itemsets
- **Support**: Frequency of a sequence in the database
- **Sequential Pattern**: Frequently occurring subsequence

### Sequential Pattern Example:

Consider a customer transaction database:
| Customer ID | Transaction Time | Items Purchased |
|-------------|------------------|----------------|
| C1          | Day 1            | A, B           |
| C1          | Day 3            | C, D           |
| C1          | Day 5            | E              |
| C2          | Day 2            | A, C           |
| C2          | Day 4            | B              |
| C2          | Day 6            | D, E           |
| C3          | Day 1            | A, B           |
| C3          | Day 3            | B, C           |
| C3          | Day 5            | D              |

Customer sequences:
- C1: ⟨(A,B), (C,D), (E)⟩
- C2: ⟨(A,C), (B), (D,E)⟩
- C3: ⟨(A,B), (B,C), (D)⟩

With minimum support of 2 customers (66%), some frequent sequential patterns include:
- ⟨(A)⟩: Supported by all 3 customers
- ⟨(A), (D)⟩: Supported by C1, C2, C3
- ⟨(A,B)⟩: Supported by C1, C3

This analysis shows that customers who purchase item A are likely to purchase item D in a later transaction.

### Applications:
- Customer purchase sequences
- Web browsing patterns
- DNA sequence analysis
- System event logs analysis

### GSP (Generalized Sequential Pattern) Algorithm:
1. Count support of each item
2. Generate candidate sequences
3. Scan database to find frequent sequences
4. Repeat steps 2-3 until no more frequent sequences found

### PrefixSpan Algorithm:
- Projects sequence database recursively
- More efficient than GSP for dense datasets
- Avoids candidate generation

### SPADE (Sequential Pattern Discovery using Equivalence classes) Algorithm:
- Uses vertical data format representation
- Employs lattice theory to decompose search space
- Reduces database scans

## 1.5 Scalable Methods for Data Mining

As datasets grow larger, traditional algorithms become inefficient. Scalable data mining methods address these challenges.

### Approaches for Scalability:
- **Sampling**: Using representative subsets of data
- **Parallelization**: Distributing computation across multiple processors
- **Distributed Computing**: Using frameworks like Hadoop, Spark
- **Dimensionality Reduction**: Decreasing the number of variables

### MapReduce for Data Mining:
- **Map Phase**: Data partitioned and processed in parallel
- **Reduce Phase**: Results combined from map tasks

### Scalable Algorithms for Pattern Mining:
- **SON (Savasere-Omiecinski-Navathe) Algorithm**: Divides dataset into chunks for parallel processing
- **Distributed FP-Growth**: Parallelized version of FP-Growth algorithm
- **MR-Apriori**: MapReduce implementation of Apriori

# 2. Types of Data

## 2.1 Categorical and Numerical Data

> Categorical Data or Qualitative Data

This type of data represents labels, categories, or names and is non-numeric.
It can be divided into:

1. **Nominal Data (Unordered Categorical Data):** Data that consists of categories without any order.
   Examples:
   - Gender: {Male, Female, Other}
   - Eye Color: {Blue, Brown, Green}
   - Blood Type: {A, B, AB, O}
1. **Ordinal Data (Ordered Categorical Data):** Categorical data with a meaningful order but **no fixed interval** between values.
   Examples:
   - Education Level: {Primary, Secondary, Higher}
   - Customer Satisfaction: {Poor, Average, Good, Excellent}
   
> Numerical Data or Quantitative Data

Numerical data represents measurable quantities and is further classified into:
1. **Discrete Data (Countable Numeric Data):** Data that can take only integer values.
   Examples:
   - Number of students in a class (30, 31, 32)
   - Number of cars in a parking lot
2. **Continuous Data (Measured Numeric Data) :** Data that can take any value within a range. 
   Examples:
   - Temperature (37.50C, 40.20C)
   - Height (5.8 ft, 6.1 ft)
   - Weight (70.5 kg, 80.2 kg)

> Other Specialized Data Types

Some data types do not fit directly into categorical or numerical categories
and require specialized handling.

1. Time-Series Data: Data collected at regular time intervals.
   Examples:
   - Stock market prices
   - Temperature variations per hour
2. Spatial Data: Data representing objects in space (geographical, topological).
   Examples:
   - GPS locations (Latitude, Longitude)
   - Map-based customer distribution

## 2.2 **ETL**: **Extract Transform Load** process.

**Examples:** Microsoft SSILS, Apache Spark  

### **Flow:** Extract Multiple Data -> Transform Data -> Load Data
- **Extract**: After verification and validation data is collected.
-  **Transform**: Proper filtering, formatting and cleaning of data and converted to a proper model.
- **Load**: Load the data in the warehouse for future decision making purposes.

### **Data Extraction Challenges**
- Complexity of data source.
- Real time data extraction.
- Volume and velocity of data.
- Access & security permissions.
- Data quality & consistency.

### **ETL vs ELT**
ETL (Extract Transform Load) (Data Warehouse) | ELT (Extract Load Transform) (Data Lake)
---|---
Performed before loading | Performed after loading
Stores processed data | Stores raw data
External ETL tools used | Cloud based warehouse used
Slower | Faster
Structured, on-premises data warehouses |Cloud-based analytics & big data

### Data Warehouse vs Data Mart vs Data Lake
Feature | Data Warehouse | Data Mart | Data Lake
---|---|---|---
Data Type | Processed and structured | Subset of data warehouse | Raw / Unstructured / Semi Structured Data
Purpose | Enterprice wise analysis and reporting | Department specific analysis | Big Data, AI, ML
Storage | High Cost optimized for better querying | Lower Cost | Cheap Cost
Example| Amazon's Sales & customer data | Amazon marketing analysis | Netflix storage for all users


---

# 3. Schemas

## 3.1 **Star Schema:**
- Star Schema is a type of data modelling technique used in data warehousing to represent data in a structured way.
- The **Fact Table** in a star schema contains the measures or metrics. (This table stores the **foreign keys** which can be found in the **Dimension Table**).
- The **Dimension Table** ins a star schema contains the descriptive attributes of the measures in the fact table.

 ### Advantages of Star Schema:
- Query performance.
- Simplicity.
- Flexibility
- Scalability. 

### Disadvantages of Star Schema:
- Time taking update.

## 3.2 **Snow Flake Schema:** 

**Schema for Multi Dimensional Data Model**
From a Star Schema, if we extend any of the Dimension table by normalizing that table, it becomes a **Snow Flake Schema**

### Advantages of Snow Flake Schema:
- Easy update.
- High Data Integrity.

### Disadvantages of Snow Flake Schema:
- Low query performance.
- Complex.

---

# 4. OLTP & OLAP

## 4.1 OLTP: Online Transaction Processing

### Use Cases:
- Retail & E-commerce.
- Banking & Finance.
- Travel & Hospitality.
- Healthcare.
- Telecommunications.
- Education.
  etc...
  
### OLTP Architecture
By the following structure the transaction happens:
1. Database Server
2. Application Server
3. User Interface


### Differences between OLTP & OLAP

OLTP (Online Transaction Processing) | OLAP (Online Analytical Processing)
--- | ---
Current Data | Historical Data
Day to day transaction operations | Data Analysis & decision making
Normalized data structures |Star or Snow Flake Schemas. Reference: [[3. Schemas]]
Simple queries | Complex queries
Used by front line employees, users,  | Used by analyst, executives, decision makers
Requires fast response time |Can have longer response times
Data updated in real time |Data periodically refreshed like once a week/month/year.

---
# 5. Data Transformation

## 5.1 Normalization in Data Transformation

##### It is used to scale and standardize the features of a dataset. 
Its primary feature is to bring all the features to a similar scale., typically between 0 and 1.

1. Min Max Normalization Technique.
2. Z - Score Normalization Technique.

### Min Max Technique:

> Formula ` X' = ( X - min_value) / (max_value - min_value)`
Here: X' -> New Value, X -> Any value.

Lets say we have the following data:

House | Sqft | Bedrooms | Price (in lakh)
--- | --- | --- | ---
1|1200|3|50
2|1500|4|60
3|1000|2|40
4|1800|5|80

By using Min Max Technique:
- Calculate by Sqft: (taking row 1):
  X' = (1200 - 1000) / (1800 - 1000) = 200 / 800 = 0.25
  Similiarly for row 2: 0.625.
  For row 3: 0
  For row 4: 1
- Calculate by bedrooms: (taking row 1):
  X' = (3 - 2) / (5 - 2) = 1 / 3 = 0.33
  ...


### Z - Score Technique:
It is also known as **Zero Mean Method**. 

> Formula ` X' = ( X - mean) / Standard Deviation`
Here: X' -> New Value, X -> Any value.

Lets say we have the following data:

Student | Height (inches)
--- | ---
1| 64
2| 70
3| 68
4| 76
5| 72

Here `mean = (64 + 70 + 68 + 76 + 72)/5 = 350 / 5 = 70`
Here Standard Deviation = 4 [Find Standard Deviation]([byjus.com/standard-deviation-formula/](https://byjus.com/standard-deviation-formula/))

So for height:
- Row 1: X' = ( 64 - 70 ) / 4 = -1.5 
- Row 2: 0
- Row 3: -0.5
- Row 4: 1.5
- Row: 5: 0.5

## 5.2. Data Discretization

It is a data preprocessing technique used to transform continuous data into discrete categories or bins.

##### Example
Calculate the bin width. If you decode to have 4 age groups, and the age range is 18 - 70, the bin width will be:
 -> Original Data: [21, 35, 28, 42, 50, 18, 70, 30, 40, 60]

ans. So group = 4
Width -> `(max - min) / 4 = (70 - 18) / 4 = 13`

- Storing Data in Bin 1:
  Range: (min, min + width -1) = (18, 30)
- Storing Data in Bin 2:
  Range ((min + width + 1),  (min + 2 x width -1)) = (31, 43)
- Storing Data in Bin 3:
  Range ((min + 2 x width +1),  (min + 3 x width -1)) = (44, 56)
- Storing Data in Bin 4: 
  Range ((min + 3 x width +1), max) = (57, 70)
  
  ##### Here we just discretized continuous data.

## 5.3. Data Preprocessing

It refers to the techniques and procedures used to prepare raw data
into a clean, organized, and structured format suitable for analysis
or modeling.
- Improved Data Quality
- Enhancing Model Performance.
- Reducing Computational Complexity
- Ensuring Compatibility.

#### Techniques:
1. Handling Missing Values: Imputation, deletion, or prediction-based methods.
2. HandIing Outliers: Trimming, or transformation-based methods.
3. Normalization/Scaling: Min-max scaling, z-score normalization, or robust scaling.
4. Encoding Categorical Variables: One-hot encoding, label encoding, or target encoding. 
5. Feature Extraction: Selecting relevant features or transforming existing features. 
6. Data Splitting: Splitting data into training, validation, and test sets for model evaluation.

---
# 6. Data Clustering

Clustering is the process o grouping similar objects together.

#### Major Clustering Methods

1. Partitioning Methods These methods divide the dataset into k partitions (or clusters), where each partition represents a cluster.
   Examples:
   - K-Means Clustering: Assigns each point to the nearest cluster centroid and iteratively updates centroids.
   - K-Medoids (PAM - Partitioning Around Medoids): Similar to K-Means but chooses actual data points as centroids, making it robust to outliers.

2. Hierarchical Methods These methods create a tree-like structure of clusters.
   Types:
   - Agglomerative (Bottom-Up): Each data point starts as an individual and merges step-by-step.
   - Divisive (Top-Down): The whole dataset starts as one cluster and splits into smaller ones.

3. Density-Based Methods Clusters are formed based on the density of data points. Key Algorithms:
   - DBSCAN (Density-Based Spatial Clustering of Applications with Noise): clusters of arbitrary shape and identifies noise (outliers).

# 7. Classification and Prediction

Classification is the process of assigning objects to predefined categories, while prediction is estimating continuous values based on patterns in the data.

## 7.1 Classification Techniques

### Decision Trees
A tree-like model where each internal node represents a feature, each branch represents a decision rule, and each leaf node represents an outcome.

**Example:** Predicting whether a customer will buy a product.
- **Root node**: Customer age
- **Branch 1**: Age < 30 → Check income
- **Branch 2**: Age ≥ 30 → Check previous purchases
- **Leaf nodes**: "Will buy" or "Won't buy"

### Naive Bayes
A probabilistic classifier based on Bayes' theorem, assuming independence between features.

**Example:** Email spam classification
- Analyzes words like "discount," "free," "urgent" for spam probability
- If P(spam|words) > P(not spam|words), classifies as spam

### Support Vector Machines (SVM)
Finds a hyperplane that best separates classes with the maximum margin.

**Example:** Credit risk assessment
- Features: income, credit history, employment status
- Hyperplane separates "low risk" from "high risk" loan applicants

### Neural Networks
Systems inspired by the human brain, with interconnected nodes (neurons) organized in layers.

**Example:** Handwritten digit recognition
- Input layer: Pixel values of a digit image
- Hidden layers: Process and transform data
- Output layer: Probabilities for each digit (0-9)

## 7.2 Prediction Techniques

### Linear Regression
Models the relationship between a dependent variable and one or more independent variables.

**Example:** House price prediction
- Features: square footage, number of bedrooms, location
- Equation: Price = 50000 + 100*sqft + 5000*bedrooms + 20000*location_score

### Time Series Forecasting
Analyzes time-ordered data points to predict future values.

**Example:** Retail sales forecasting
- Historical sales data shows 15% increase every December
- Model predicts similar pattern for upcoming December

## 7.3 Evaluation Metrics

### For Classification
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### For Prediction
- **Mean Absolute Error (MAE)**: Average of absolute differences between predictions and actual values
- **Root Mean Square Error (RMSE)**: Square root of the average of squared differences
- **R-squared**: Proportion of variance in the dependent variable explained by the model

# 8. Cluster Analysis

## 8.1 Types of Data in Cluster Analysis

### Interval-Scaled Variables
Variables measured on a scale with equal intervals, where both order and difference are meaningful.

**Example:** Temperature measurements (20°C is exactly 5°C higher than 15°C)

### Binary Variables
Variables with only two possible values (0/1, yes/no, true/false).

**Example:** Customer purchase status (bought/didn't buy)

### Nominal Variables
Categorical variables without any meaningful order.

**Example:** Colors of products (red, blue, green)

### Ordinal Variables
Categorical variables with a meaningful order but no consistent scale.

**Example:** Customer satisfaction ratings (poor, good, excellent)

### Ratio-Scaled Variables
Variables with meaningful zero points and ratios.

**Example:** Age, income, weight

### Mixed Variables
Datasets containing multiple types of variables.

**Example:** Customer profiles with age (ratio), gender (binary), and preferences (nominal)

## 8.2 Partitioning Methods

### K-Means Clustering
Divides data into k clusters where each observation belongs to the cluster with the nearest mean.

**Example:** Customer segmentation
- Customers grouped by purchase amount and frequency
- Cluster 1: High spenders, frequent shoppers
- Cluster 2: High spenders, infrequent shoppers
- Cluster 3: Low spenders, frequent shoppers
- Cluster 4: Low spenders, infrequent shoppers

### K-Medoids (PAM)
Similar to K-means but uses actual data points (medoids) as cluster centers instead of means.

**Example:** Location planning for service centers
- Customer locations plotted on a map
- Service centers placed at medoid locations (actual customer addresses)
- More robust to outliers than K-means

## 8.3 Hierarchical Methods

### Agglomerative (Bottom-Up) Approach
Starts with each observation as a separate cluster and merges the closest clusters until only one remains.

**Example:** Document clustering
1. Start with each document as its own cluster
2. Merge most similar documents (e.g., similar topics)
3. Continue merging until reaching desired number of clusters or threshold

### Divisive (Top-Down) Approach
Starts with all observations in one cluster and recursively splits into smaller clusters.

**Example:** Market segmentation
1. Start with all customers in one group
2. Split based on major differences (e.g., age)
3. Further split each group (e.g., by income)
4. Continue until reaching desired granularity

## 8.4 Transactional Patterns and Temporal-Based Frequent Patterns

### Cyclic Patterns
Regular repeating patterns in transaction data.

**Example:** Retail purchasing cycles
- Ice cream sales spike every summer
- Heater sales increase every winter
- Holiday decorations sell well before specific holidays

### Seasonal Patterns
Patterns related to seasons or fixed periods.

**Example:** Travel bookings
- Beach resort bookings increase in summer
- Ski resort bookings increase in winter
- Business hotel bookings decrease on weekends

### Calendar-Based Patterns
Patterns that follow calendar events.

**Example:** Restaurant traffic
- Higher dinner sales on Friday/Saturday
- Higher lunch sales on weekdays
- Lower overall sales on Mondays

# 9. Mining Time Series Data

## 9.1 Periodicity Analysis for Time-Related Sequence Data

Periodicity analysis identifies regular repeating patterns in time series data.

### Types of Periodicity
- **Full Periodicity**: The entire sequence repeats
- **Partial Periodicity**: Only some patterns repeat
- **Cyclic Periodicity**: Irregular intervals between repetitions

**Example:** Retail store traffic
- Daily periodicity: Peak hours at lunch and after work
- Weekly periodicity: Higher traffic on weekends
- Annual periodicity: Holiday shopping season peaks

## 9.2 Trend Analysis

Trend analysis identifies long-term movements in time series data.

### Types of Trends
- **Linear Trend**: Consistent growth or decline
- **Exponential Trend**: Growth/decline at an increasing rate
- **Seasonal Trend**: Regular fluctuations around a trend line

**Example:** Smartphone sales over time
- Upward trend as technology adoption increases
- Seasonal spikes with new model releases
- Temporary dips during economic downturns

## 9.3 Similarity Search in Time-Series Analysis

Finding patterns in time series data that match or closely resemble a query pattern.

### Distance Measures
- **Euclidean Distance**: Direct point-to-point comparison
- **Dynamic Time Warping (DTW)**: Flexible matching allowing time shifts
- **Correlation-based Distance**: Measures pattern similarity regardless of scale

**Example:** Stock market pattern matching
- Identifying historical patterns similar to current market conditions
- Comparing company performance across different time periods
- Finding correlations between different stock movements

# 10. Mining Data Streams

## 10.1 Methodologies for Stream Data Processing

### Window-Based Processing
Processing data using sliding or tumbling windows of fixed size.

**Example:** Social media sentiment analysis
- Analyze last 1000 tweets about a product
- Window slides forward as new tweets arrive
- Oldest tweets exit the window

### Sampling-Based Approaches
Selecting representative subsets of the stream for analysis.

**Example:** Network traffic monitoring
- Sample every 10th packet for analysis
- Maintain statistical properties while reducing processing load

### Sketching Algorithms
Creating compact summaries of data streams.

**Example:** Website visitor counting
- Count-Min Sketch tracks approximate frequency of visitors
- Uses much less memory than storing full visitor logs

## 10.2 Frequent Pattern Mining in Stream Data

### Lossy Counting Algorithm
Approximates frequent itemsets in a data stream with bounded error.

**Example:** Online retail transaction stream
- Track frequently purchased item combinations
- Automatically forget old or infrequent patterns
- Update store recommendations in real-time

## 10.3 Sequential Pattern Mining in Data Streams

### Sliding Window Sequential Pattern Mining
Discovers sequential patterns within a moving window of recent events.

**Example:** User clickstream analysis
- Identify common navigation paths through a website
- Patterns like: Homepage → Product Category → Specific Product → Cart
- Adapt website layout based on discovered patterns

## 10.4 Classification of Dynamic Data Streams

### Hoeffding Trees (Very Fast Decision Trees)
Incrementally builds a decision tree for streaming data.

**Example:** Credit card fraud detection
- Model learns from transaction stream
- Updates fraud patterns immediately with new data
- Adapts to emerging fraud techniques without retraining

## 10.5 Class Imbalance Problem

When one class significantly outnumbers others, creating challenges for classification algorithms.

**Example:** Intrusion detection
- Normal network traffic (99.9%) vs. malicious traffic (0.1%)
- Models might classify everything as normal to achieve high accuracy
- Solutions: Oversampling minority class, undersampling majority class, or using cost-sensitive learning

## 10.6 Graph Mining

Analyzing data represented as graphs with nodes and edges.

### Community Detection
Identifying groups of nodes that are densely connected internally but sparsely connected with the rest of the network.

**Example:** Social network friend circles
- Finding groups of people who frequently interact
- Identifying isolated communities within larger networks

### Frequent Subgraph Mining
Discovering subgraph patterns that appear frequently in a graph dataset.

**Example:** Chemical compound analysis
- Identifying common molecular structures in a database of chemical compounds
- Finding structural motifs associated with certain properties

## 10.7 Social Network Analysis

### Centrality Measures
Identifying important nodes in a network.

**Example:** Influencer identification
- Degree centrality: Person with most connections
- Betweenness centrality: Person connecting different groups
- Eigenvector centrality: Person connected to other important people

### Link Prediction
Predicting future connections between nodes.

**Example:** Friend recommendations
- Suggesting connections based on common friends
- Predicting potential business partnerships based on industry connections

# 11. Web Mining

## 11.1 Mining Web Page Layout Structure

Analyzing the structure and organization of web pages to extract useful information.

**Example:** Automated content extraction
- Distinguishing navigation menus from main content
- Identifying headings, paragraphs, and lists
- Extracting product information from e-commerce sites

## 11.2 Mining Web Link Structure

Analyzing the hyperlinks between web pages to determine importance and relationships.

### PageRank Algorithm
Measures the importance of web pages based on the number and quality of links to them.

**Example:** Search engine ranking
- Pages linked from many important pages rank higher
- Links from authoritative sites carry more weight

## 11.3 Mining Multimedia Data on the Web

Extracting information from images, videos, and audio files on the web.

**Example:** Content-based image retrieval
- Searching for visually similar images
- Categorizing images by content (landscapes, portraits, etc.)
- Identifying products in images for shopping recommendations

## 11.4 Automatic Classification of Web Documents

Categorizing web pages into predefined classes.

**Example:** News article categorization
- Classifying articles as sports, politics, technology, etc.
- Using text content, metadata, and link information
- Supporting personalized content recommendations

## 11.5 Web Usage Mining

Analyzing web user behavior and interaction patterns.

**Example:** Website optimization
- Identifying common navigation paths
- Detecting pages with high bounce rates
- Optimizing conversion funnels based on user behavior

# 12. Distributed Data Mining

## 12.1 Distributed Warehousing

Storing and managing data across multiple locations or servers.

**Example:** Global retail chain data management
- Sales data stored in regional data centers
- Centralized access for global analytics
- Local processing for regional reports

## 12.2 Recent Trends in Distributed Data Mining

### Grid Computing
Connecting multiple computers to work together on complex tasks.

**Example:** Climate model analysis
- Dividing complex calculations across hundreds of computers
- Combining results for comprehensive analysis

### Cloud-Based Data Mining
Using cloud infrastructure for scalable data mining.

**Example:** Netflix recommendation system
- Processing viewing habits of millions of users
- Elastically scaling resources based on demand
- Distributed storage and processing across global data centers

### Edge Computing
Processing data near its source rather than in a centralized location.

**Example:** Smart factory monitoring
- Sensors process data locally to detect equipment failures
- Only relevant insights sent to central systems
- Reduced latency for time-critical decisions

### Federated Learning
Training machine learning models across multiple devices without sharing the underlying data.

**Example:** Mobile keyboard prediction
- Models trained locally on users' typing patterns
- Only model updates shared with central system
- Preserves privacy while improving predictions
