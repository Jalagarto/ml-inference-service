
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Input(BaseModel):
    features: list[float]


class Output(BaseModel):
    prediction: float


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: Input):
    weights = [2.5, -1.2, 0.8]
    result = sum(f * w for f, w in zip(data.features, weights))
    return Output(prediction=result)
