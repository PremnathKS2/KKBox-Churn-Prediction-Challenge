# Steps
1.	Data Analysis
2.	Data Cleansing
3.	Feature Engineering
4.	Model Identification
5.	Train and validate the model 
6.	Predict the price for test.tsv data using the trained model

## Data Analysis
### A.	Numerical features
•	
•	payment_plan_days : How many days plan 25, 50, 100 etc...

### B.	Categorical features
•	is_auto_renew : A binary indicator, 1 if auto renewval is true and 0 if auto renewval false.
•	gender : Specifying about member gender
•	payment_method_id : Which payment method used to subscribe
•	is_cancel : subcriber can cancel or not
•	
•	

There are 1048576 unique subscribers data in the members data
And 970961 unique records in train data with is_churn values

Filtered sample data in sample_submission_v2 within the date range of Apr. To June 2017

#### is_churn(Target Variable)
is_churn: A binary indicator, is_churn = 1 means churn,is_churn = 0 means renewal.
This is the target variable, Churn is defined as weather the user did not continued the subscription with in 30 Days of expiration.

## Cleansing Data
1. Finding the null datetime column
2. Filling the Registration Date_time with valid data in train and test data

## Feature Engineering
1. All the data are valid format and information featured, and not required to additional formatting.

## Model Identification
1. Identified the category of the problem statement (Regression)
2. Identified 2-3 algorithms to fulfill the requirement (Logistic Regression,Decision Classifier,Neural Network Classifier)
3. Compared (Logistic, Decision and Neural ) based on performance - and decided to go ahead with Decision Classifier which is showing more accuracy
  
## Train and Validate the model



# Potential shortcomings
1. Identifying independent Variables, This means that logistic regression is not a useful tool unless researchers have already identified all the relevant independent variables
2. Independent Observation Required. 
3. The amount of computational power needed for a Neural Network depends heavily on the size of your data but also on how deep and complex your Network is

# Possible improvements
1. Increasing the accuracy of neural network classification using refined training data
