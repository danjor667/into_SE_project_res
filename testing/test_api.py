import requests

url = "http://localhost:8000/predict/"

data = {"Pregnancies": 10,
        "Glucose": 182,
        "BloodPressure": 84,
        "SkinThickness": 40,
        "Insulin":105,
        "BMI":33.6,
        "DiabetesPedigreeFunction": 0.501,
        "Age": 20
        }



response = requests.post(url, data)
print(response.json())