import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Create a simple dataset
data = {
    "years_experience": [1, 2, 3, 4, 5, 6 ],
    "salary": [30000, 35000, 40000, 45000, 50000, 55000]
}

df = pd.DataFrame(data)

# 2. Split features and target
X = df[["years_experience"]]
y = df["salary"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make prediction
years = [[7]]
predicted_salary = model.predict(years)

print(f"Predicted salary for 7 years experience: {predicted_salary[0]:.2f}")

# 6. Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X), linestyle='--')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Linear Regression")
plt.show()