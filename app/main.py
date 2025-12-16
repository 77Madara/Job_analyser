from fastapi import  FastApi
from models import Text
from models import summarizer, nlp, classifier

app = FastApi()




@app.post("/analyze")
async def analyze(text: Text):

    return text