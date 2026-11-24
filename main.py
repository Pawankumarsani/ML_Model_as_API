import io
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app =FastAPI(
    title="California House Price Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

model = joblib.load("House_Model.joblib")
features = joblib.load("House_Features.joblib")

class HouseFeatures(BaseModel):
    MedInc : float = Field(gt=0, description="Median Income of Neighbourhood")
    HouseAge : float = Field(ge=0, description="Average age of House")
    AveRooms : float = Field(gt=0, description="Average numbers of rooms")
    AveBedrms : float = Field(gt=0, description="Average numbers of bedrooms")
    Population : float = Field(gt=0, description="Total Population")
    AveOccup : float = Field(gt=0, description="Average numbers of occupation")
    Latitude : float = Field(ge=32, le=42, description="Latitude")
    Longitude : float = Field(ge=-125, le=-114, description="Longitude")
    
@app.get("/")
def home():
    return {
        "Message": "California House Prediction API",
        "Status": "running",
        "Endpoint": "send POST request to /predict"
    }
    
@app.get("/health")
def health():
    return {
        "Status": "Running",
        "Model": "RandomForestRegressor",
        "features": features,
        "avg_errors": "$32773"
    }

@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc":house.MedInc,
            "HouseAge":house.HouseAge,
            "AveRooms":house.AveRooms,
            "AveBedrms":house.AveBedrms,
            "Population":house.Population,
            "AveOccup":house.AveOccup,
            "Latitude":house.Latitude,
            "Longitude":house.Longitude,
        }])
        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000
        
        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"${predicted:,.2f} hundred thousands",
            "fidance_range": f"${price_usd - 32773:,.0f} to ${price_usd + 32773:,.0f}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed: {str(e)}"
        )

@app.post("/predict-file")
async def predict_file(file: UploadFile=File(...)):
    
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="please upload a CSV file only"
            )

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the CSV file"
            )

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"These columns are missing from your file: {missing}"
            )
        
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file has no data rows"
            )

    try:
        predictions = model.predict(df[list(features)])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
            )

    df["predicted_price_usd"] = predictions * 100000
    df["predicted_price_usd"] = df["predicted_price_usd"].apply(lambda x: f"${x:,.0f}")

    return StreamingResponse(
        io.StringIO(df.to_csv(index=False)),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )