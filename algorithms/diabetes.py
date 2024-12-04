import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


# Define input data structure
class InputData(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int

def scaleInput(df):
    with open('saved_models/robust_scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)

    scaled_df = pd.DataFrame(loaded_scaler.transform(df), columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'], index = [0])
    return scaled_df

def processInput(data: InputData):
    # Handle the missing data case - use the same approach as in the notebook to fill it with medians
    NewBMI = pd.Series(["Underweight","Normal", "Overweight","Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")
    df = pd.DataFrame({
        "Pregnancies": [data.pregnancies],
        "Glucose": [data.glucose],
        "BloodPressure": [data.blood_pressure],
        "SkinThickness": [data.skin_thickness],
        "Insulin": [data.insulin],
        "BMI": [data.bmi],
        "DiabetesPedigreeFunction": [data.diabetes_pedigree_function],
        "Age": [data.age]
    })
    scaled_df = scaleInput(df)

    # Add categorical variables
    categorical_columns = {'NewBMI_Obesity 1': False,
       'NewBMI_Obesity 2': False, 'NewBMI_Obesity 3': False, 'NewBMI_Overweight': False,
       'NewBMI_Underweight': False, 'NewInsulinScore_Normal': False, 'NewGlucose_Low': False,
       'NewGlucose_Normal': False, 'NewGlucose_Overweight': False, 'NewGlucose_Secret': False}

    df = df.assign(**categorical_columns)
  
    # Update these categorical variables
    if df['BMI'].iloc[0] < 18.5:
        df['NewBMI_Underweight'] = True
    elif df['BMI'].iloc[0] > 24.9 and df['BMI'].iloc[0] <= 29.9:
        df['NewBMI_Overweight'] = True
    elif df['BMI'].iloc[0] > 29.9 and df['BMI'].iloc[0] <= 34.9:
        df['NewBMI_Obesity 1'] = True
    elif df['BMI'].iloc[0] > 34.9 and df['BMI'].iloc[0] <= 39.9:
        df['NewBMI_Obesity 2'] = True
    elif df['BMI'].iloc[0] > 39.9:
        df['NewBMI_Obesity 3'] = True 
    
    if df['Insulin'].iloc[0] >= 16 and df['Insulin'].iloc[0] <= 166:
        df['NewInsulinScore_Normal'] = True

    if df['Glucose'].iloc[0] <= 70:
        df['NewGlucose_Low'] = True
    elif df['Glucose'].iloc[0] > 70 and df['Glucose'].iloc[0] <= 99:
        df['NewGlucose_Normal'] = True
    elif df['Glucose'].iloc[0] > 99 and df['Glucose'].iloc[0] <= 126:
        df['NewGlucose_Overweight'] = True
    elif df['Glucose'].iloc[0] > 126:
        df['NewGlucose_Secret'] = True
    
    columns_to_shift = ['NewBMI_Obesity 1',
       'NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight',
       'NewBMI_Underweight', 'NewInsulinScore_Normal', 'NewGlucose_Low',
       'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']
    for col in columns_to_shift:
       scaled_df[col] = df[col].iloc[0]

    with open('saved_models/standard_scaler.pkl', 'rb') as f:
        standard_scaler = pickle.load(f)
     
    scaled_df = standard_scaler.transform(scaled_df) 

    return scaled_df

# Initialize FastAPI
app = FastAPI()

# Load the model (make sure the file path is correct)
svm_model = joblib.load('saved_models/svc_model.pkl')

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Prepare input data for prediction
    processed_input = processInput(data)

    # Predict using the SVM model
    prediction = svm_model.predict(processed_input)

    # Return the prediction
    return {"prediction": int(prediction[0])}
