import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Generate synthetic sentiment dataset
data = {
    'Review': ["Amazing product", "Worst experience", "I love it", "Not worth the money", "Highly recommended"],
    'Sentiment': [1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# Tokenization
tokenizer = Tokenizer(num_words=100)  # Limiting vocab size
tokenizer.fit_on_texts(df['Review'])
sequences = tokenizer.texts_to_sequences(df['Review'])

# Padding sequences to ensure consistent input shape
max_length = max(len(seq) for seq in sequences)  # Get max sequence length
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = np.array(df['Sentiment'])

# Define LSTM Model
model = Sequential([
    Embedding(input_dim=100, output_dim=10, input_length=max_length),
    LSTM(10),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model (for demo, no validation)
model.fit(X, y, epochs=10, batch_size=2)
