import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. dataset
data = {
    "text": [
        "I love this product",
        "This is amazing",
        "I am very happy",
        "I hate this",
        "This is terrible",
        "Very bad experience"
    ],
    "sentiment": [1, 1, 1, 0, 0, 0]  # 1 = positive, 0 = negative
}

df = pd.DataFrame(data)

# 2. split X and y
X = df["text"]
y = df["sentiment"]

# 3. convert text to numbers
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# 4. train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# 5. model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. predict new text
new_text = ["I really love this"]
new_text_vec = vectorizer.transform(new_text)
prediction = model.predict(new_text_vec)

if prediction[0] == 1:
    print("Positive sentiment :)")
else:
    print("Negative sentiment :(")