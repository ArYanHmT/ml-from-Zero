import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# dataset (fake customers)
data = {
    "Age": [18, 19, 20, 22, 25, 45, 46, 48, 50, 52],
    "Spending": [100, 120, 130, 150, 170, 800, 820, 900, 950, 1000]
}

df = pd.DataFrame(data)

# features
X = df[["Age", "Spending"]]

# create model
model = KMeans(n_clusters=2)

# train
model.fit(X)

# get cluster labels
labels = model.labels_

# plot
plt.scatter(df["Age"], df["Spending"], c=labels)
plt.xlabel("Age")
plt.ylabel("Spending")
plt.title("Customer Segmentation")
plt.show()