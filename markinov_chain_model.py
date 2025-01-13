import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
n_states = len(cleaned_data['Previous_Move'].unique())  
transition_matrix = np.zeros((n_states, n_states))  
for (prev_move, curr_move) in zip(X_train['Previous_Move'], y_train):
    transition_matrix[prev_move, curr_move] += 1
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
print("Transition Matrix:")
print(transition_matrix)
def predict_next_move(last_move, transition_matrix):

    if np.isnan(transition_matrix[last_move]).all():  
        return np.random.choice(range(len(transition_matrix)))
    return np.argmax(transition_matrix[last_move])
y_pred = []
for i in range(len(X_test)):
    last_move = X_test.iloc[i]['Previous_Move']
    y_pred.append(predict_next_move(last_move, transition_matrix))

y_pred = np.array(y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy using Markov Chain Model: {accuracy:.2f}")
