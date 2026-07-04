 
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib, numpy as np

artifact = joblib.load("../service/nova_pay_fraud_model_lean.joblib")                           # load the LEAN model artifact (short feature list)
model    = artifact["model"]                                        # the sklearn model object itself
FEATURES = artifact["feature_cols"]                                 # the model's own feature list (short form)
THRESH   = artifact["threshold"]                                    # the model's own threshold for classifying fraud

app = FastAPI(title="Nova Pay Fraud Scoring API", version="2.0")    # app title and version for the OpenAPI docs

# Pydantic BaseModel built from the model's OWN feature list (lean form).
class Transaction(BaseModel):
    txn_velocity_1h: float
    txn_velocity_24h: float
    ip_risk_score: float
    device_trust_score: float
    country_location_mismatch: int = Field(ge=0, le=1)
    amount_usd: float                                                      # dynamically built from the model's own feature list, so the request form always matches whatever the model expects

class ScoreResponse(BaseModel):
    fraud_probability: float                                        # probability of fraud (0.0-1.0)
    is_fraud: bool                                                  # whether the transaction is classified as fraud (True/False)
    threshold: float                                                # the threshold used for classification (0.0-1.0)

@app.get("/health")                                                 # health check endpoint for monitoring
def health(): 
    return {"status": "ok", "n_features": len(FEATURES), "features": FEATURES}    # Return health status and feature information

@app.post("/score", response_model=ScoreResponse)                                   # score endpoint for scoring transactions
def score(txn: Transaction):
    row = np.array([[getattr(txn, f) for f in FEATURES]])                           # build a 2D array (1 row, n_features columns) from the Transaction object, in the order of FEATURES
    prob = float(model.predict_proba(row)[0, 1])   # [0,1] = P(fraud) for row 0     # define the probability of fraud for the transaction using the model's predict_proba method
    return ScoreResponse(fraud_probability=round(prob, 4),                          # round to 4 decimal places
                         is_fraud=prob >= THRESH, threshold=THRESH)                 # return the scoring results
