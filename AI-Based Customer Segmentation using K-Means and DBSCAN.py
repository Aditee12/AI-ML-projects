import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Generate synthetic customer data
np.random.seed(42)
num_samples = 300

age = np.random.randint(18, 70, num_samples)
income = np.random.randint(20000, 120000, num_samples)
purchase_frequency = np.random.randint(1, 20, num_samples)
spending_score = np.random.randint(1, 100, num_samples)

data = pd.DataFrame({'Age': age, 'Income': income, 'Purchase_Frequency': purchase_frequency, 'Spending_Score': spending_score})

# Normalize data for clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(data_scaled)

# Apply DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
data['DBSCAN_Cluster'] = dbscan.fit_predict(data_scaled)

# Visualization of Clusters (K-Means)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(data['Income'], data['Spending_Score'], c=data['KMeans_Cluster'], cmap='viridis')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('K-Means Clustering')

# Visualization of Clusters (DBSCAN)
plt.subplot(1, 2, 2)
plt.scatter(data['Income'], data['Spending_Score'], c=data['DBSCAN_Cluster'], cmap='coolwarm')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('DBSCAN Clustering')

plt.show()