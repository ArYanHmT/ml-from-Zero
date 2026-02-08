import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# load data
df = pd.read_csv("train.csv")

# فقط ستون‌های عددی
df = df.select_dtypes(include=["int64", "float64"])

# حذف سطرهایی که مقدار خالی دارن
df = df.dropna()

# target
y = df["SalePrice"]

# features
X = df.drop("SalePrice", axis=1)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LinearRegression()

# train
model.fit(X_train, y_train)

# predict
preds = model.predict(X_test)

# error
mae = mean_absolute_error(y_test, preds)

print("MAE:", mae)