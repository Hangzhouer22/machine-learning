#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Background


# In[ ]:


#McCurr Health Consultancy is an MNC that has thousands of employees spread across the globe. 
#The company believes in hiring the best talent available and retaining them for as long as possible. 
#A huge amount of resources is spent on retaining existing employees through various initiatives. 
#The Head of People Operations wants to bring down the cost of retaining employees. For this, he proposes 
#limiting the incentives to only those employees who are at risk of attrition. As a recently hired Data 
#Scientist in the People Operations Department, you have been asked to identify patterns in characteristics 
#of employees who leave the organization. Also, you have to use this information to predict if an employee 
#is at risk of attrition. This information will be used to target them with incentives.


# In[5]:


#Objective


# In[ ]:


#To identify the different factors that drive attrition
#To build a model to predict if an employee will attrite or not


# In[3]:


#Dataset Description


# In[4]:


#The data contains information on employees' demographic details, work-related metrics, and attrition flag.

#EmployeeNumber - Unique Employee Identifier
#Attrition - Did the employee attrite or not?
#Age - Age of the employee
#BusinessTravel - Travel commitments for the job
#DailyRate - Data description not available
#Department - Employee's Department
#DistanceFromHome - Distance from work to home (in KM)
#Education - Employee's Education. 1-Below College, 2-College, 3-Bachelor, 4-Master, 5-Doctor
#EducationField - Field of Education
#EnvironmentSatisfaction - 1-Low, 2-Medium, 3-High, 4-Very High
#Gender - Employee's gender
#HourlyRate - Data description not available
#JobInvolvement - 1-Low, 2-Medium, 3-High, 4-Very High
#JobLevel - Level of job (1 to 5)
#JobRole - Job Roles
#JobSatisfaction - 1-Low, 2-Medium, 3-High, 4-Very High
#MaritalStatus - Marital Status
#MonthlyIncome - Monthly Salary
#MonthlyRate - Data description not available
#NumCompaniesWorked - Number of companies worked at
#Over18 - Whether the employee is over 18 years of age?
#OverTime - Whether the employee is doing overtime?
#PercentSalaryHike - The percentage increase in the salary last year
#PerformanceRating - 1-Low, 2-Good, 3-Excellent, 4-Outstanding
#RelationshipSatisfaction - 1-Low, 2-Medium, 3-High, 4-Very High
#StandardHours - Standard Hours
#StockOptionLevel - Stock Option Level
#TotalWorkingYears - Total years worked
#TrainingTimesLastYear - Number of training attended last year
#WorkLifeBalance - 1-Low, 2-Good, 3-Excellent, 4-Outstanding
#YearsAtCompany - Years at Company
#YearsInCurrentRole - Years in the current role
#YearsSinceLastPromotion - Years since the last promotion
#YearsWithCurrManager - Years with the current manager
#In the real world, you will not find definitions for some of your variables. 
#It is the part of the analysis to figure out what they might mean.

#Note
#Kindly do not run the code cells containing Hyperparameter Tuning using GridSearchCV during the session, 
#since they take considerable time to run.


# In[2]:


#Importing the libraries and overview of the dataset


# In[6]:


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# To scale the data using z-score
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

# Algorithms to use
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

# Metrics to evaluate the model
from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve,recall_score

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

# For tuning the model
from sklearn.model_selection import GridSearchCV

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[7]:


#Loading the Dataset


# In[8]:


# Loading the dataset
df = pd.read_excel('/Users/yutaoyan/Desktop/HRemployeeGL/HREmployee.xlsx')


# In[9]:


# Looking at the first 5 records
df.head()


# In[10]:


#Checking the info of the dataset


# In[11]:


# Let us see the info of the data
df.info()


# In[12]:


#Observations:

#There are 2940 observations and 34 columns in the dataset.
#All the columns have 2940 non-null values, i.e., there are no missing values in the data.


# In[13]:


#Let's check the unique values in each column


# In[14]:


# Checking the count of unique values in each column
df.nunique()


# In[15]:


#Observations:

#Employee number is an identifier which is unique for each employee and we can drop this column as 
#it would not add any value to our analysis.
#Over18 and StandardHours have only 1 unique value. These columns will not add any value to our model 
#hence we can drop them.
#Over18 and StandardHours have only 1 unique value. We can drop these columns as they will not add any 
#value to our analysis.
#On the basis of number of unique values in each column and the data description, we can identify the 
#continuous and categorical columns in the data.


#Let's drop the columns mentioned above and define lists for numerical and categorical columns to 
#explore them separately.


# In[16]:


# Dropping the columns
df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours'] , axis = 1)


# In[17]:


# Creating numerical columns
num_cols = ['DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'TotalWorkingYears', 
            'YearsAtCompany', 'NumCompaniesWorked', 'HourlyRate', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
            'YearsWithCurrManager', 'TrainingTimesLastYear']

# Creating categorical variables
cat_cols = ['Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 'EducationField', 'JobSatisfaction', 'EnvironmentSatisfaction', 
            'WorkLifeBalance', 'StockOptionLevel', 'Gender', 'PerformanceRating', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus', 'RelationshipSatisfaction']


# In[56]:


df2 = df.groupby('Attrition').median()


# In[57]:


df2[num_cols].style.highlight_max(color="lightgreen")


# In[59]:


df = df.drop(['PercentSalaryHike', 'YearsSinceLastPromotion','PerformanceRating', 'HourlyRate'], axis=1)


# In[18]:


#univariate analysis and data preprocessing and move to the model building section.


# In[19]:


#Univariate analysis of numerical columns


# In[20]:


# Checking summary statistics
df[num_cols].describe().T


# In[21]:


#Observations:

#Average employee age is around 37 years. It has a high range, from 18 years to 60, indicating good age 
#diversity in the organization.
#At least 50% of the employees live within a 7 KM radius of the organization. However, there are some 
#extreme values, given that the maximum value is 29 km.
#The average monthly income of an employee is USD 6500. It has a high range of values from 1K-20K USD, 
#which is to be expected for any organization's income distribution. There is a big difference between
#the 3rd quartile value (around USD 8400) and the maximum value (nearly USD 20000), showing that the 
#company's highest earners have a disproportionately large income in comparison to the rest of the 
#employees. Again, this is fairly common in most organizations.
#The average salary hike of an employee is around 15%. At least 50% of employees got a salary hike of
#14% or less, with the maximum salary hike being 25%.
#The average number of years an employee is associated with the company is 7.
#On average, the number of years since an employee got a promotion is ~2.19. The majority of employees
#have been promoted since the last year.


# In[22]:


# Creating histograms
df[num_cols].hist(figsize = (14, 14))

plt.show()


# In[23]:


#Observations:

#The age distribution is close to a normal distribution, with the majority of employees between the ages
#of 25 and 50.
#DistanceFromHome also has a right-skewed distribution, meaning most employees live close to work but there 
#are a few that live further away.
#MonthlyIncome and TotalWorkingYears are skewed to the right, indicating that the majority of workers are in 
#entry / mid-level positions in the organization.
#The percentage salary hike is skewed to the right, which means employees are mostly getting lower percentage 
#salary increaseS.
#The YearsAtCompany variable distribution shows a good proportion of workers with 10+ years, indicating a 
#significant number of loyal employees at the organization.
#The YearsInCurrentRole distribution has three peaks at 0, 2, and 7. There are a few employees that have even 
#stayed in the same role for 15 years and more.
#The YearsSinceLastPromotion variable distribution indicates that some employees have not received a promotion
#in 10-15 years and are still working in the organization. These employees are assumed to be high work-experience 
#employees in upper-management roles, such as co-founders, C-suite employees, etc.
#The distributions of DailyRate, HourlyRate, and MonthlyRate appear to be uniform and do not provide much 
#information. It could be that the daily rate refers to the income earned per extra day worked while the hourly
#rate could refer to the same concept applied for extra hours worked per day. Since these rates tend to be 
#broadly similar for multiple employees in the same department, that explains the uniform distribution they show.


# In[24]:


#Univariate analysis for categorical variables


# In[25]:


# Printing the % sub categories of each category.
for i in cat_cols:
    
    print(df[i].value_counts(normalize = True))
    
    print('*' * 40)


# In[26]:


#Observations:
#The employee attrition rate is ~16%.
#Around 28% of the employees are working overtime. This number appears to be on the higher side and might
#indicate a stressed employee work-life.
#71% of the employees have traveled rarely, while around 19% have to travel frequently.
#Around 73% of the employees come from an educational background in the Life Sciences and Medical fields.
#Over 65% of employees work in the Research & Development department of the organization.
#Nearly 40% of the employees have low (1) or medium-low (2) job satisfaction and environment satisfaction 
#in the organization, indicating that the morale of the company appears to be somewhat low.
#Over 30% of the employees show low (1) to medium-low (2) job involvement.
#Over 80% of the employees either have none or very few stock options.
#In terms of performance ratings, none of the employees have rated lower than 3 (excellent).
#About 85% of employees have a performance rating equal to 3 (excellent), while the remaining have a 
#rating of 4 (outstanding). This could either mean that the majority of employees are top performers, 
#or the more likely scenario is that the organization could be highly lenient with its performance appraisal 
#process.


# In[27]:


#Model Building - Approach
#Data preparation.
#Partition the data into a train and test set.
#Build a model on the train data.
#Tune the model if required.
#Test the data on the test set.


# In[28]:


#Data preparation


# In[29]:


#Creating dummy variables for the categorical variables


# In[30]:


# Creating a list of columns for which we will create dummy variables
to_get_dummies_for = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'MaritalStatus', 'JobRole']

# Creating dummy variables
df = pd.get_dummies(data = df, columns = to_get_dummies_for, drop_first = True)      

# Mapping overtime and attrition
dict_OverTime = {'Yes': 1, 'No': 0}
dict_attrition = {'Yes': 1, 'No': 0}

df['OverTime'] = df.OverTime.map(dict_OverTime)
df['Attrition'] = df.Attrition.map(dict_attrition)


# In[ ]:


#Separating the independent variables (X) and the dependent variable (Y)


# In[31]:


# Separating the target variable and other variables

Y = df.Attrition

X = df.drop(['Attrition'], axis = 1)


# In[32]:


#Splitting the data into 70% train and 30% test set


# In[33]:


# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1, stratify = Y)


# In[34]:


# Creating metric function

def metrics_score(actual, predicted):
    
    print(classification_report(actual, predicted))
    
    cm = confusion_matrix(actual, predicted)
    
    plt.figure(figsize = (8, 5))
    
    sns.heatmap(cm, annot = True, fmt = '.2f', xticklabels = ['Not Attriate', 'Attriate'], yticklabels = ['Not Attriate', 'Attriate'])
    
    plt.ylabel('Actual')
    
    plt.xlabel('Predicted')
    
    plt.show()


# In[35]:


# Building decision tree model
dt = DecisionTreeClassifier(class_weight = {0: 0.17, 1: 0.83}, random_state = 1)


# In[36]:


# Fitting decision tree model
dt.fit(x_train, y_train)


# In[37]:


# Checking performance on the training dataset
y_train_pred_dt = dt.predict(x_train)

metrics_score(y_train, y_train_pred_dt)


# In[38]:


#Observation:

#The Decision tree is giving a 100% score for all metrics on the training dataset.


# In[39]:


# Checking performance on the test dataset
y_test_pred_dt = dt.predict(x_test)

metrics_score(y_test, y_test_pred_dt)


# In[60]:


np.round(dt.feature_importances_, 4)


# In[40]:


# Plot the feature importance

importances = dt.feature_importances_

columns = X.columns

importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)

plt.figure(figsize = (13, 13))

sns.barplot(importance_df.Importance,importance_df.index)


# In[41]:


# Choose the type of classifier
dtree_estimator = DecisionTreeClassifier(class_weight = {0: 0.17, 1: 0.83}, random_state = 1)

# Grid of parameters to choose from
parameters = {'max_depth': np.arange(4,5,6,7), 
              'criterion': ['gini', 'entropy'],
              'min_samples_leaf': [5, 10, 20, 25]
             }

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(recall_score, pos_label = 1)

# Run the grid search
gridCV = GridSearchCV(dtree_estimator, parameters, scoring = scorer, cv = 10)

# Fitting the grid search on the train data
gridCV = gridCV.fit(x_train, y_train)

# Set the classifier to the best combination of parameters
dtree_estimator = gridCV.best_estimator_

# Fit the best estimator to the data
dtree_estimator.fit(x_train, y_train)


# In[61]:


gridCV.best_params_


# In[42]:


# Checking performance on the training dataset
y_train_pred_dt = dtree_estimator.predict(x_train)

metrics_score(y_train, y_train_pred_dt)


# In[43]:


# Checking performance on the test dataset
y_test_pred_dt = dtree_estimator.predict(x_test)

metrics_score(y_test, y_test_pred_dt)


# In[44]:


importances = dtree_estimator.feature_importances_

columns = X.columns

importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)

plt.figure(figsize = (13, 13))

sns.barplot(importance_df.Importance, importance_df.index)


# In[45]:


features = list(X.columns)

plt.figure(figsize = (30, 20))

tree.plot_tree(dt, max_depth = 4, feature_names = features, filled = True, fontsize = 12, node_ids = True, class_names = True)

plt.show()


# #Fitting the Random Forest classifier on the training data
# rf_estimator = RandomForestClassifier(n_estimators=500, class_weight = "balanced", random_state = 1, max_depth=2)
# 
# rf_estimator.fit(x_train, y_train)

# In[64]:


# Fitting the Random Forest classifier on the training data
rf_estimator = RandomForestClassifier(n_estimators=500, class_weight = "balanced", random_state = 1, max_depth=2)
rf_estimator.fit(x_train, y_train)


# In[47]:


# Checking performance on the training data
y_pred_train_rf = rf_estimator.predict(x_train)

metrics_score(y_train, y_pred_train_rf)


# In[48]:


# Checking performance on the testing data
y_pred_test_rf = rf_estimator.predict(x_test)

metrics_score(y_test, y_pred_test_rf)


# In[49]:


importances = rf_estimator.feature_importances_

columns = X.columns

importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)

plt.figure(figsize = (13, 13))

sns.barplot(importance_df.Importance, importance_df.index)


# In[69]:


# Choose the type of classifier
rf_estimator_tuned = RandomForestClassifier(class_weight ="balanced", random_state = 1)

# Grid of parameters to choose from
params_rf = { "max_depth": [2,3,4,5,6],
        "n_estimators": [100, 250, 500],
        "min_samples_leaf": np.arange(1, 4, 1),
        "max_features": ['log2', 'auto'],
}


# Type of scoring used to compare parameter combinations - recall score for class 1
scorer = "recall"

# Run the grid search
grid_obj = GridSearchCV(rf_estimator_tuned, params_rf, scoring = "recall", cv = 3, n_jobs=-1)

grid_obj2 = grid_obj.fit(x_train, y_train)

# Set the classifier to the best combination of parameters
rf_estimator_tuned = grid_obj2.best_estimator_


# In[66]:


rf_estimator_tuned.fit(x_train, y_train)


# In[67]:


# Checking performance on the training data
y_pred_train_rf_tuned = rf_estimator_tuned.predict(x_train)

metrics_score(y_train, y_pred_train_rf_tuned)


# In[68]:


# Plotting feature importance
importances = rf_estimator_tuned.feature_importances_

columns = X.columns

importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)

plt.figure(figsize = (13, 13))

sns.barplot(importance_df.Importance, importance_df.index)


# In[ ]:




