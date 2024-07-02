from rest_framework import mixins, generics
from load_model import MODEL, SCALER



class  DiabetesPredictor(generics.GenericAPIView):
    def post(self, request):
        pass