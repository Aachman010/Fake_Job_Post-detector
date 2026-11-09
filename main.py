from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
import pandas as pd
import joblib
from pathlib import Path

# Initialize FastAPI
app = FastAPI(
    title="Fake Job Posting Detector",
    description="API to detect fraudulent job postings using NLP and Machine Learning",
    version="1.0.0"
)

# Load the trained ML pipeline
MODEL_PATH = Path(__file__).parent / "fake_job_post_detector_model.pkl"
# Load the saved model and threshold
saved_model = joblib.load(MODEL_PATH)

model = saved_model['pipeline']
THRESHOLD = saved_model['threshold']


# Input schema
class JobPosting(BaseModel):
    text: str=Field(min_length=20)

    location: str = ""
    employment_type: str = ""
    required_experience: str = ""
    required_education: str = ""
    industry: str = ""
    function: str = ""

    telecommuting: int = 0
    has_company_logo: int = 0
    has_questions: int = 0


# Home endpoint
@app.get("/")
def home():
    return FileResponse('index.html')

# Prediction endpoint
@app.post("/predict")
def predict_job(data: JobPosting):

    try:
        # Calculate numerical text features
        text_length = len(data.text)
        word_count = len(data.text.split())

        # Convert input into DataFrame
        input_data = pd.DataFrame([{
            "text": data.text,
            "location": data.location,
            "employment_type": data.employment_type,
            "required_experience": data.required_experience,
            "required_education": data.required_education,
            "industry": data.industry,
            "function": data.function,
            "telecommuting": data.telecommuting,
            "has_company_logo": data.has_company_logo,
            "has_questions": data.has_questions,
            "text_length": text_length,
            "word_count": word_count
        }])

        # Get probability of fraud (class 1)
        fraud_probability = model.predict_proba(input_data)[0][1]

        # Apply custom threshold
        prediction = int(fraud_probability >= THRESHOLD)

        return {
            "prediction": prediction,
            "result": "Fraudulent" if prediction == 1 else "Genuine",
            "fraud_probability": round(float(fraud_probability), 4),
            "threshold": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))