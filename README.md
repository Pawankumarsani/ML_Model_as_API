# California Housing Price Prediction

A machine learning project that predicts house prices using the California
Housing dataset. The model is trained with `RandomForestRegressor`, saved with
`joblib`, and served through a FastAPI REST API. The whole app runs in Docker.

## Files

| File | What it does |
| --- | --- |
| `train.py` | Trains the Random Forest model and saves the `.joblib` files |
| `main.py` | FastAPI app that loads the model and serves predictions |
| `test_house.csv` | Sample data used to test predictions |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Builds the container image |
| `House_Model.joblib` | Saved trained model (created by `train.py`) |
| `House_Features.joblib` | Saved list of feature names (created by `train.py`) |

## Dataset

The project uses the California Housing dataset that comes with `scikit-learn`,
so you don't need to download anything. It has 20,640 rows, 8 input features and
one target value.

Features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`,
`AveOccup`, `Latitude`, `Longitude`.

Target: `MedHouseVal` — median house value, in units of $100,000 (so `2.5`
means about $250,000).

## Setup

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

This creates two files:

- `House_Model.joblib` — the trained Random Forest model
- `House_Features.joblib` — the feature names, in the order the model expects

Run this again whenever you change the model, then restart the API.

## Run the API

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API docs.

## API

### `GET /`

Checks that the API is running.

### `POST /predict`

Send the 8 features as JSON:

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

Response:

```json
{ "predicted_price": 4.526 }
```

The value is in units of $100,000, so `4.526` is about $452,600.

## Testing

**Postman:** create a `POST` request to `http://127.0.0.1:8000/predict`, set
Body → raw → JSON, paste the JSON above, and send it. You should get `200 OK`.

**Swagger UI:** open `/docs`, expand `POST /predict`, click *Try it out*, paste
a row from `test_house.csv`, and click *Execute*.

## Docker

```bash
docker build -t house-price-api .
docker run -d --name house-price-api -p 8000:8000 house-price-api
```

The API is then at `http://127.0.0.1:8000`.

Handy commands:

```bash
docker logs house-price-api     # see logs
docker stop house-price-api     # stop the container
```

Make sure `House_Model.joblib` and `House_Features.joblib` exist before
`docker build`, and that the Dockerfile copies them into the image. Otherwise
the container starts but predictions fail.

## Note

The model is trained on 1990 US census data and predicts the median value for a
block group, not an individual house. Treat the output as a demo, not a real
price estimate.
