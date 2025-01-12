from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np


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


rf = RandomForestClassifier(random_state=42)


param_grid = {
    'n_estimators': [50, 100, 200],     
    'max_depth': [None, 10, 20, 30],   
    'min_samples_split': [2, 5, 10],   
    'min_samples_leaf': [1, 2, 4]     
}


grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search_rf.fit(X_train, y_train)

best_rf = grid_search_rf.best_estimator_
print(f"Best Hyperparameters: {grid_search_rf.best_params_}")

y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
