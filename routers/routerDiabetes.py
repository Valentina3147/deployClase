import pickle
from fastapi import APIRouter
from Diabetesdata import Diabetesdata
from schemas import schemas
import numpy as np

router=APIRouter()

pkl_filenamec="RFDiabetesv102.pkl"
with open(pkl_filenamec,'rb') as file:
    model = pickle.load(file)

labels=["sano","Posible diabetes"]



@router.get("/")
async def root():
    return{
        "message":"AI Service"
    }

@router.post("/Predict")
def Predict_diabetes(data:schemas.Diabetesdata):
    data = data.model_dump()

    Pregnancies = data["Pregnancies"]
    glucosa = data["glucosa"]
    bloodpresure = data["bloodpresure"]
    skinthickness = data["skinthickness"]
    insuline = data["insuline"]
    BMI = data["BMI"]
    diabetes = data["diabetes"]
    age = data["age"]

    xin = np.array([
        Pregnancies,
        glucosa,
        bloodpresure,
        skinthickness,
        insuline,
        BMI,
        diabetes,
        age]).reshape(1,8)

    prediction = model.predict(xin)
    yout = labels[prediction[0]]

    return{
        'prediction':yout
    }