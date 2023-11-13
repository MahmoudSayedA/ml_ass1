# Assignment 1: Linear and Logistic Regression

A housing finance company offers interest-free home loans to customers.
When a customer applies for a home loan, the company validates the
customer's eligibility for a loan before making a decision.
Now, the company wants to automate
the customers' eligibility validation
process based on the customers' details
provided while filling the application
form. These details include gender,
education, income, credit history, and
others. The company also wants to have
a predictive model for the maximum
loan amount that an applicant is
authorized to borrow based on his details.
You are required to build a linear regression model and a logistic regression
model for this company to predict loan decisions and amounts based on
some features.

## Datasets:

There are two attached datasets:

● The first dataset “loan_old.csv” contains 614 records of applicants' data
with 10 feature columns in addition to 2 target columns. The features
are: the loan application ID, the applicant's gender, marital status,
number of dependents, education and income, the co-applicant's
income, the number of months until the loan is due, the applicant's
credit history check, and the property area. The targets are the
maximum loan amount (in thousands) and the loan acceptance status.

● The second dataset “loan_new.csv” contains 367 records of new
applicants' data with the 10 feature columns.


## Requirements:

Write a Python program in which you do the following:
a) Load the "loan_old.csv" dataset.
b) Perform analysis on the dataset to:
- check whether there are missing values
- check the type of each feature (categorical or numerical)
- check whether numerical features have the same scale
- visualize a pairplot between numercial columns
c) Preprocess the data such that:
- records containing missing values are removed
- the features and targets are separated
- the data is shuffled and split into training and testing sets
- categorical features are encoded
- categorical targets are encoded
- numerical features are standardized
d) Fit a linear regression model to the data to predict the loan amount.
-> Use sklearn's linear regression.
e) Evaluate the linear regression model using sklearn's R^2 score.
f) Fit a logistic regression model to the data to predict the loan status.
- Implement logistic regression from scratch using gradient descent.
g) Write a function (from scratch) to calculate the accuracy of the model.
h) Load the "loan_new.csv" dataset.
i) Perform the same preprocessing on it (except shuffling and splitting).
j) Use your models on this data to predict the loan amounts and status.

