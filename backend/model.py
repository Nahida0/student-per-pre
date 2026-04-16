import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 2, 1, 6, 7, 3, 4, 5],
    'attendance':  [40, 50, 60, 70, 80, 85, 90, 95, 45, 30, 88, 92, 65, 75, 78],
    'prev_grade':  [30, 40, 50, 60, 70, 75, 80, 90, 35, 25, 82, 88, 55, 65, 72],
    'result':      [0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  1,  1,  0,  1,  1]
}

df = pd.DataFrame(data)
X = df[['study_hours', 'attendance', 'prev_grade']]
y = df['result']

model = LogisticRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("model.pkl created successfully!")