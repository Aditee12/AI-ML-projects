import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Generate synthetic movie rating dataset
data = {
    'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'User1': [5, 3, 4, 4, 2],
    'User2': [4, 2, 5, 3, 3],
    'User3': [3, 5, 4, 2, 4],
    'User4': [4, 3, 5, 4, 2]
}

df = pd.DataFrame(data)

# Compute similarity
similarity_matrix = cosine_similarity(df.iloc[:, 1:])

# Print similarity matrix with movie names as row/column labels
print(pd.DataFrame(similarity_matrix, index=df['Movie'], columns=df['Movie']))
