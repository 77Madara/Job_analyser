from fastapi import  FastAPI
from models import ProfileResponse, ProfileRequest
from analysis import analyze_profile


app = FastAPI(title="AI Profile Analyzer")




@app.post("/analyze-profile", response_model=ProfileResponse)
async def analyze(request: ProfileRequest):
    result = analyze_profile(request.text)

    return result