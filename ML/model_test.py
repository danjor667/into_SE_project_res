import pickle
import numpy as np
import pandas as pd

with open("new_model.pkl", "rb") as file:
    MODEL = pickle.load(file)


with open("scaler.pkl", "rb") as file2:
    scaler = pickle.load(file2)


X = [
    [9,119,80,35,0,29,0.263,29],
    [11,143,94,33,146,36.6,0.254,51],
    [10,125,70,26,115,31.1,0.205,41],
    [4,127,80,0,0,44.4,0.257,13],
    ["4", "110", "89", "34", "98", "36.6", "0.154", "25"]
]

features_names = ["Pregnancies",
                  "Glucose",
                  "BloodPressure",
                  "SkinThickness",
                  "Insulin",
                  "BMI",
                  "DiabetesPedigreeFunction",
                  "Age"
                  ]

X_new = np.array(X)

df = pd.DataFrame(X_new, columns=features_names)
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=features_names)


prob = MODEL.decision_function(df_scaled)

prediction1 = MODEL.predict(df_scaled)
print(prob)


print(prediction1)


