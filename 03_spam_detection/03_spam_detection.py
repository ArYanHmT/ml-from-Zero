import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. dataset
data = {
    "email": [
        "Win a free iPhone now",
        "Congratulations you won a prize",
        "Claim your free gift",
        "Meeting at 6 pm",
        "Let's have lunch tomorrow",
        "Call me when you get home"
    ],
    "label": [1, 1, 1, 0, 0, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# 2. split X and y
X = df["email"]
y = df["label"]

# 3. convert text to numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# 4. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# 5. model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. test with new email
new_email = ["You won a free ticket"]
new_email_vec = vectorizer.transform(new_email)

prediction = model.predict(new_email_vec)

if prediction[0] == 1:
    print("Spam ")
else:
    print("Not Spam ")