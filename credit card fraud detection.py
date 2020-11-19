#!/usr/bin/env python
# coding: utf-8

# # 1. Credit Card Fraud Detection Intuitions
# 
# What is Credit Card Fraud?
# Credit card fraud is when someone uses another person's credit card or account information to make unauthorized purchases or access funds through cash advances. Credit card fraud doesn’t just happen online; it happens in brick-and-mortar stores, too. As a business owner, you can avoid serious headaches – and unwanted publicity – by recognizing potentially fraudulent use of credit cards in your payment environment.
# 
# Problem Statement:
# The Credit Card Fraud Detection Problem includes modeling past credit card transactions with the knowledge of the ones that turned out to be a fraud. This model is then used to identify whether a new transaction is fraudulent or not. Our aim here is to detect 100% of the fraudulent transactions while minimizing the incorrect fraud classifications.
# 
# Observations
# Very few transactions are actually fraudulent (less than 1%). The data set is highly skewed, consisting of 492 frauds in a total of 284,807 observations. This resulted in only 0.172% fraud cases. This skewed set is justified by the low number of fraudulent transactions.
# The dataset consists of numerical values from the 28 ‘Principal Component Analysis (PCA)’ transformed features, namely V1 to V28. Furthermore, there is no metadata about the original features provided, so pre-analysis or feature study could not be done.
# The ‘Time’ and ‘Amount’ features are not transformed data.
# There is no missing value in the dataset.
# 
# Why does class imbalanced affect model performance?
# In general, we want to maximize the recall while capping FPR (False Positive Rate), but you can classify a lot of charges wrong and still maintain a low FPR because you have a large number of true negatives.
# 
# Business questions to brainstorm:
# Since all features are anonymous, we will focus our analysis on non-anonymized features: Time, Amount
# 
# How different is the amount of money used in different transaction classes?
# Do fraudulent transactions occur more often during a certain frames?
# 

# In[1]:


##Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


##Importing data
data = pd.read_csv('C:/Users/sourabh/Desktop/credit card fraud detection/creditcard.csv')
data.head()


# In[4]:


data['Class'].value_counts()


# In[6]:


fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]
print(fraud.shape)
print(normal.shape)


# In[7]:


##EDA
sns.relplot(x='Time', y='Amount',data=fraud,kind='scatter',hue='Amount')
plt.show()


# In[8]:


fraud.corr()
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
data.columns


# In[9]:


X = data.drop('Class', axis=1)
y = data['Class']


# In[15]:


##UP SAMPLING
smk = SMOTETomek(random_state=42)
X_res, y_res = smk.fit_sample(X,y)
X_res.shape


# In[16]:


y_res.shape


# In[17]:


##Data PreProcessing
scaling = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

X_train = scaling.fit_transform(X_train)
X_test = scaling.transform(X_test)


# In[18]:


def print_score(label, prediction, train=True):
    if train:
        clf_report = pd.DataFrame(classification_report(label, prediction, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Classification Report:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, prediction)}\n")
        
    elif train==False:
        clf_report = pd.DataFrame(classification_report(label, prediction, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Classification Report:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(label, prediction)}\n")


# In[19]:


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

y_train_pred = rf_clf.predict(X_train)
y_test_pred = rf_clf.predict(X_test)

print_score(y_train, y_train_pred, train=True)
print_score(y_test, y_test_pred, train=False)


# In[ ]:




