import numpy as np
from mistune import HTMLRenderer
from rest_framework import mixins, generics, status
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from .load_model import MODEL, SCALER
import pandas as pd



class  DiabetesPredictor(APIView):
    renderer_classes = [JSONRenderer]
    def post(self, request, *args, **kwargs):
        try:
            data = request['POST']
            print("this i sthe data i got **********************")
            print(data)
            features_names = ["Pregnancies",
                              "Glucose",
                              "BloodPressure",
                              "SkinThickness",
                              "Insulin",
                              "BMI",
                              "DiabetesPedigreeFunction",
                              "Age"
                              ]
            # for name in features_names:
            #     features[name] = data.get(name, 0)
            X = [data]
            X = pd.DataFrame(X, columns=features_names)
            print(X)

            scaled_X = SCALER.fit_transform(X)
            prediction = MODEL.predict(scaled_X)
            probability = MODEL.decision_function(scaled_X)
            print("this one is call")
            return Response({"test":"testing"})
            # return Response({"prediction": prediction[0]}, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"this is the problem ************************** {e}")
            return Response({"error": "an error"}, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, *args, **kwargs):
        return Response({"text": "you are not making the correct request"}, status=status.HTTP_200_OK)

predict_view = DiabetesPredictor.as_view()