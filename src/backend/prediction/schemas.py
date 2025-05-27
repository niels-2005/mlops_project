from pydantic import BaseModel


class PredictionInput(BaseModel):
    age: int
    sex: int
    chest_pain_type: int
    resting_bp_s: int
    cholesterol: int
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: int
    exercise_angina: int
    oldpeak: float
    st_slope: int


class PredictionResponseData(BaseModel):
    output: int
    output_proba: float
