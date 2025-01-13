import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, silhouette_score

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

# Naive Bayes 


nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)


y_pred_nb = nb_model.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print(f"Accuracy using Naive Bayes: {accuracy_nb:.2f}")
print(classification_report(y_test, y_pred_nb))

#  K-Means 


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

silhouette_avg = silhouette_score(X, kmeans_labels)
print("\n=== K-Means Clustering ===")
print(f"Silhouette Score: {silhouette_avg:.2f}")

cleaned_data['Cluster_Label'] = kmeans_labels


X_cluster = cleaned_data[['Age', 'Gender', 'Previous_Move', 'Cluster_Label']]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cluster, y, test_size=0.2, random_state=42)
nb_model.fit(X_train_c, y_train_c)
y_pred_cluster = nb_model.predict(X_test_c)
accuracy_cluster = accuracy_score(y_test_c, y_pred_cluster)
print(f"Naive Bayes with Cluster Labels Accuracy: {accuracy_cluster:.2f}")

# ---------------------- K-Nearest Neighbor ----------------------

knn_model = KNeighborsClassifier(n_neighbors=3)  
knn_model.fit(X_train, y_train)


y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("\n=== K-Nearest Neighbor (KNN) ===")
print(f"Accuracy using KNN: {accuracy_knn:.2f}")
print(classification_report(y_test, y_pred_knn))
