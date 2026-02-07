import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. dataset
data = {
    "email": [
        "Win a free iPhone now",
        "Congratulations you won a prize",
        "Claim your free gift",
        "Meeting at 6 pm",
        "Let's have lunch tomorrow",
        "Call me when you get home",
        "Free money waiting for you",
        "Urgent: Your account is at risk",
        "Don't miss this offer",
        "How are you today?",
        "Can we meet tomorrow?",
        "Dinner at my place tonight?"
    ],
    "label": [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]  # 1 = spam, 0 = ham
}
df = pd.DataFrame(data)

# 2. split
X = df["email"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 3. vectorize text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 5. predict
y_pred = model.predict(X_test_vec)

# 6. evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# 7. confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)