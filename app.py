from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

#Templates
templates = Jinja2Templates(directory="templates")

with open("30-diamond_model_complete.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    encoders = saved_data["encoders"]
    scaler = saved_data["scaler"]

class DiamondFeatures(BaseModel):
    carat : float
    cut : str
    color : str
    clarity : str
    depth : float
    table : float
    x : float
    y : float
    z : float

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/predict")
async def predict(features: DiamondFeatures):
    try:
        input_data = pd.DataFrame([features.model_dump()])

        # 🔥 HARDCODED SAFE MAPPINGS
        cut_map = {
            "Fair": 0,
            "Good": 1,
            "Very Good": 2,
            "Premium": 3,
            "Ideal": 4
        }

        color_map = {
            "J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6
        }

        clarity_map = {
            "I1": 0, "SI2": 1, "SI1": 2,
            "VS2": 3, "VS1": 4,
            "VVS2": 5, "VVS1": 6, "IF": 7
        }

        # 🔥 DIRECT NUMERIC CONVERSION (encoder yok)
        input_data.at[0, "cut"] = cut_map[input_data.at[0, "cut"]]
        input_data.at[0, "color"] = color_map[input_data.at[0, "color"]]
        input_data.at[0, "clarity"] = clarity_map[input_data.at[0, "clarity"]]

        # kolon sırası
        input_data = input_data[
            ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
        ]

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        return {"predicted_price": float(prediction[0])}

    except Exception as e:
        print("❌ PREDICT ERROR:", e)
        return {"error": str(e)}

