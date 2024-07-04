import requests

url = "http://127.0.0.1:8000/predict/"

data = {"Pregnancies": 4,
        "Glucose": 110,
        "BloodPressure": 89,
        "SkinThickness": 34,
        "Insulin":98,
        "BMI":36.6,
        "DiabetesPedigreeFunction": 0.154,
        "Age": 25
        }



response = requests.post(url, data)
print(response.json())