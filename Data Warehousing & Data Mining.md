# 1. Introduction

## Types of **Data**

> Categorial Data or Qualitative Data

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

## **ETL**: **Extract Transform Load** process.

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

# 2. Schemas

## **Star Schema:**
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

## **Snow Flake Schema:** 

**Schema for Multi Dimensional Data Model**
From a Star Schema, if we extend any of the Dimension table by normalizing that table, it becomes a **Snow Flake Schema**

### Advantages of Snow Flake Schema:
- Easy update.
- High Data Integrity.

### Disadvantages of Snow Flake Schema:
- Low query performance.
- Complex.

---

# 3. OLTP & OLAP

## OLTP: Online Transaction Processing

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
Normalized data structures |Star or Snow Flake Schemas. Reference: [[2. Schemas]]
Simple queries | Complex queries
Used by front line employees, users,  | Used by analyst, executives, decision makers
Requires fast response time |Can have longer response times
Data updated in real time |Data periodically refreshed like once a week/month/year.

---
# 4. Data Transformation

## 4.1 Normalization in Data Transformation

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

## 4.2. Data Discretization

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

## 4.3. Data Preprocessing

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
