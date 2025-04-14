import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate synthetic news dataset
data = {
    "News": [
        "Breaking news! You won a lottery", 
        "Stock market crashes",
        "Exclusive offer for you", 
        "Government announces new policy", 
        "Win a free trip"
    ],
    "Label": [1, 0, 1, 0, 1]  # 1 = Fake, 0 = Real
}

df = pd.DataFrame(data)
print("Dataset:\n", df)  # Debugging output

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['News'])
y = df['Label']

print("TF-IDF Matrix Shape:", X.shape)  # Debugging output
print("TF-IDF Feature Names:", vectorizer.get_feature_names_out())  # Debugging output

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Data Shape:", X_train.shape)  # Debugging output
print("Testing Data Shape:", X_test.shape)  # Debugging output

model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)
print("Predictions:", predictions)
