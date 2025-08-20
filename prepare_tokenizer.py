import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Load your dataset
df = pd.read_csv("Reviews.csv")

# Use the column containing text (adjust if different)
texts = df['Text'].astype(str).tolist()

# Create and fit tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Save tokenizer to a file
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer saved as tokenizer.pkl")
