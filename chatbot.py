import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Generate synthetic chatbot dataset
data = {
    'Question': ["What are your store hours?", "How to return a product?", "Do you offer free shipping?",
                 "Where is my order?", "What payment methods do you accept?"],
    'Answer': ["Our store hours are 9 AM - 9 PM.", "You can return within 30 days.",
               "Yes, free shipping on orders above $50.", "Track order in your account.",
               "We accept Visa, MasterCard, and PayPal."]
}

df = pd.DataFrame(data)

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Question'])
y = np.arange(len(df))  # Assigning labels to responses

# Train Model
model = SVC(kernel='linear')
model.fit(X, y)

# Chatbot Function
def chatbot_response(query):
    query_vec = vectorizer.transform([query])
    response_idx = model.predict(query_vec)[0]
    return df['Answer'][response_idx]

# Test Chatbot
test_query = "How do I return a product?"
print("User:", test_query)
print("Bot:", chatbot_response(test_query))

print("Chatbot Model Built Successfully!")
