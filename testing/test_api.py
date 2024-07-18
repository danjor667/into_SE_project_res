import requests

url = "http://into-se-project-res.onrender.com/predict/"

data = {"Pregnancies": 10,
        "Glucose": 132,
        "BloodPressure": 84,
        "SkinThickness": 40,
        "Insulin":105,
        "BMI":33.6,
        "DiabetesPedigreeFunction": 0.201,
        "Age": 20
        }



response = requests.post(url, data)
print(response.json())