import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate synthetic transaction data
np.random.seed(42)
transactions = {
    'Amount': np.random.randint(10, 500, 1000).tolist() + np.random.randint(5000, 10000, 10).tolist(),
    'Transaction_Time': np.random.randint(1, 24, 1010),
    'Is_Fraudulent': [0] * 1000 + [1] * 10  # Marking last 10 as fraud
}

df = pd.DataFrame(transactions)

# Apply Anomaly Detection
model = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly'] = model.fit_predict(df[['Amount', 'Transaction_Time']])

# Visualization
plt.scatter(df['Amount'], df['Transaction_Time'], c=df['Anomaly'], cmap='coolwarm')
plt.xlabel("Transaction Amount")
plt.ylabel("Transaction Time")
plt.title("Fraud Detection")
plt.show()
