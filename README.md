Medical Insurance Statistical Analysis
 This application analyzes the Medical Insurance Costs dataset and demonstrates descriptive statistics, hypothesis testing, multiple linear regression, and residual diagnostics.

The applicaiton dashboard contains 3 interactive sections:
    1. Data Exploration
    2. Hypothesis Testing Lab
    3. Prediction & Diagnostics

The dataset used for this application is Medical Insurance Costs.
    The dataset contains both numerical and categorical variables. The main target variable used for regression and prediction is charges.

Features:
Feature	   Description
age	       Age of the individual
sex	       Gender of the individual
bmi	       Body Mass Index
children   Number of children/dependents
smoker	   Whether the individual is a smoker
region	   Residential region
charges	   Medical insurance cost

Technologies Used
Python
Pandas
NumPy
SciPy
Scikit-learn
Statsmodels
Plotly
Streamlit
Matplotlib
Seaborn

Statistical Analysis Performed
1. Descriptive Analysis
2. Hypothesis Testing
    Two statistical hypothesis tests are included.
    Test 1: Two-Group Comparison
    The following tests are performed:
    -Shapiro-Wilk test for normality
    -Levene's test for equality of variances
    -Independent two-sample t-test when assumptions are satisfied
    -Mann-Whitney U test when normality is not satisfied
At a significance level of α = 0.05, the application reports whether the null hypothesis is rejected or not rejected
    Test 2: One-Way ANOVA
    One-way ANOVA is used to determine whether the mean of a selected numerical variable differs significantly across the four residential regions.
    The result is interpreted using a significance level of α = 0.05
3.Multiple Linear Regression
4. Model Diagnostics

Instructions to run the application:
Install Required Packages:
    pip install -r requirements.txt
Run the Streamlit Application:
    streamlit run app.py