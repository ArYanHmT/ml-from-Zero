import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. create dataset
data = {
    "study_hours": [1, 2, 3, 4, 5, 6],
    "passed": [0, 0, 0, 1, 1, 1]  # 0 = fail, 1 = pass
}

df = pd.DataFrame(data)

# 2. split X and y
X = df[["study_hours"]]
y = df["passed"]

# 3. train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. create model
model = LogisticRegression()

# 5. train model
model.fit(X_train, y_train)

# 6. predict
hours = [[3.5]]
prediction = model.predict(hours)

if prediction[0] == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")