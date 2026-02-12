from flask import Flask, request, jsonify
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -------------------------------
# 1️⃣ مدل رو اینجا می‌سازیم
# -------------------------------

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# -------------------------------
# 2️⃣ ساخت Flask API
# -------------------------------

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    img = data["image"]  # فرض می‌کنیم لیست 64 عدد
    prediction = model.predict([img])[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)