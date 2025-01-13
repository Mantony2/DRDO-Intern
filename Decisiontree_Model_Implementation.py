from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import streamlit as st



dataset = pd.read_excel(r"C:\Users\User\OneDrive\Desktop\Processsed_DATASET.xlsx")

cleaned_data = dataset.dropna(how='all', axis=1).dropna(how='all').reset_index(drop=True)
cleaned_data.columns = cleaned_data.columns.str.strip()


encoder = LabelEncoder()

cleaned_data['Gender'] = encoder.fit_transform(cleaned_data['Gender'])  
cleaned_data['Previous_Move'] = encoder.fit_transform(cleaned_data['Previous_Move'])  
cleaned_data['Current_Move'] = encoder.fit_transform(cleaned_data['Current_Move'])  


X = cleaned_data[['Age', 'Gender', 'Previous_Move']]  
y = cleaned_data['Current_Move']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)

# GridSearch CV 
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],  
    'max_depth': [None, 5, 10, 15, 20],  
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4], 
}


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)


final_model = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")


y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy of Decision tree model with Best Hyperparameters: {accuracy:.2f}")

