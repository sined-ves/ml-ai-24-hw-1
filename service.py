from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
from io import StringIO
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np



model_pipeline = joblib.load('car_price_prediction_pipeline.pkl')

app = FastAPI()


class Car(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class CarResponse(Car):
    selling_price: float

def kmpkg_to_kmpl(value):
    if not pd.isna(value):
      if value.endswith('km/kg'):
          numeric_value = float(value.replace('km/kg', '').strip())
          return f'{numeric_value * 1.4} kmpl'
    return value

def preprocess_data(df):
    df = df.drop(['torque'], axis=1)
    df['name'] = df['name'].apply(lambda x: x.split(' ')[0])
    df['mileage'] = df['mileage'].apply(kmpkg_to_kmpl)
    df['mileage'] = df['mileage'].str.replace(' kmpl', '').astype(float)
    df['engine'] = df['engine'].str.replace(' CC', '').astype(int)
    df['max_power'] = df['max_power'].str.replace(' bhp', '').replace('', np.nan).astype(float)
    df['seats'] = df['seats'].astype(int)
    return df

@app.post("/predict_item", response_model=CarResponse)
async def predict_car_price(item: Car) -> float:
    df = pd.DataFrame([item.model_dump()])
    df = preprocess_data(df)
    prediction = round(float(model_pipeline.predict(df)[0]), 3)

    response = item.model_dump()
    response.update({'selling_price': prediction})
    return response


@app.post("/predict_items")
async def predict_car_prices_from_csv(file: UploadFile = File(...)):
    contents = await file.read()
    csv_data = StringIO(contents.decode('utf-8'))

    df = pd.read_csv(csv_data)
    df_preprocessed = preprocess_data(df)

    predictions = model_pipeline.predict(df_preprocessed)
    df['selling_price'] = predictions
    df['selling_price'] = df['selling_price'].round(3)

    output_csv = StringIO()
    df.to_csv(output_csv, index=False)
    output_csv.seek(0)

    return StreamingResponse(output_csv,
                             media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=predictions.csv"})
