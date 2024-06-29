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
    [7,147,76,0,0,39.4,0.257,43]
]

features_names = ["Pregnancies",
                  "Glucose",
                  "BloodPressure",
                  "SkinThickness",
                  "Insulin",
                  "BMI",
                  "DiabetesPedigreeFunction",
                  "Age"]

X_new = np.array(X)

df = pd.DataFrame(X_new, columns=features_names)
df_scaled = scaler.fit_transform(df)
print(pd.DataFrame(df_scaled).head())

prob = MODEL.decision_function(df_scaled)

prediction1 = MODEL.predict(df_scaled)
print(prob)


print(prediction1)


