import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Load the model (make sure the file path is correct)
svm_model = joblib.load('saved_models/svc_model.pkl')

# Define input data structure
class InputData(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    # Add all necessary features

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Prepare input data for prediction
    ip = [[-0.126039  , -0.75805027,  0.29726317, -1.6968127 , -1.42534509,
       -1.39558371, -0.73863998, -1.04098645,  0.        ,  0.36070975,
       -0.36070975,  0.        ,  0.        ,  0.72638708, -0.10792245,
        1.81154358, -0.77731695, -0.78004206]]
    input_features = np.array([[data.feature_1, data.feature_2, data.feature_3]])  # Adjust as needed
    
    # Predict using the SVM model
    prediction = svm_model.predict(ip)
    
    # Return the prediction
    return {"prediction": int(prediction[0])}

