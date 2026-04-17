from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# Training data directly as arrays (no pandas needed)
X = np.array([
    [1, 40, 30],
    [2, 50, 40],
    [3, 60, 50],
    [4, 70, 60],
    [5, 80, 70],
    [6, 85, 75],
    [7, 90, 80],
    [8, 95, 90],
    [2, 45, 35],
    [1, 30, 25],
    [6, 88, 82],
    [7, 92, 88],
    [3, 65, 55],
    [4, 75, 65],
    [5, 78, 72]
])

y = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("model.pkl created successfully!")