import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic house data
np.random.seed(42)

data = {
    'Area': np.random.randint(800, 5000, 100),
    'Bedrooms': np.random.randint(1, 5, 100),  # Fixed key name
    'Price': np.random.randint(50000, 500000, 100)
}

df = pd.DataFrame(data)  # Fixed DataFrame creation

# Train Linear Regression Model
X = df[['Area', 'Bedrooms']]
y = df['Price']

model = LinearRegression()  # Fixed model instantiation
model.fit(X, y)

# Visualization
plt.scatter(df['Area'], df['Price'], color='blue', label="Actual Prices")
plt.plot(df['Area'], model.predict(X), color='red', label="Predicted Prices")

plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show()  # Fixed plt.show()