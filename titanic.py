#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Load the dataset

# Step 1: Import necessary libraries
import pandas as pd

# Step 2: Load the dataset
df = pd.read_csv('train.csv')

# Step 3: Check the data structure
print(df.head())
print(df.info())


# In[13]:


# Data Cleaning and Preprocessing

# Step 4: Print columns to check if 'Embarked' is present
print("Columns in the dataset before processing:", df.columns)

# Step 5: Handle missing values only if 'Embarked' exists
if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Handle missing values for 'Age'
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())


# In[15]:


# Step 6: Convert categorical columns into numerical using One-Hot Encoding
if 'Sex' in df.columns:
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
if 'Embarked' in df.columns:
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# In[16]:


# Step 7: Drop unnecessary columns if they exist
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Step 8: Check the processed DataFrame
print(df.head())


# In[17]:


# Split the Data into Training and Testing Sets

# Step 1: Check if 'Survived' exists in the DataFrame before splitting the data
if '2urvived' in df.columns:
    # Step 2: Define features (X) and target (y)
    X = df.drop('2urvived', axis=1)  # Drop 'Survived' from features
    y = df['2urvived']               # Target variable
    print("'Survived' column found and split defined.")
    
    # Step 3: Split the data into training and testing sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training set size:", X_train.shape)
    print("Testing set size:", X_test.shape)
else:
    print("Error: 'Survived' column not found in the dataset. Check if it was dropped earlier.")


# In[18]:


# Apply Two Supervised Learning Algorithms
#Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 8: Apply Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred_lr = lr_model.predict(X_test)

# Step 10: Evaluate Logistic Regression
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# In[19]:


# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

# Step 11: Apply Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Step 12: Make predictions
y_pred_rf = rf_model.predict(X_test)

# Step 13: Evaluate Random Forest
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# In[11]:


# Compare and Contrast the Algorithms

# Comparison of Logistic Regression and Random Forest results
print("Comparison of the Models:")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")


# In[ ]:




