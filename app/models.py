from pydantic import BaseModel

class Text(BaseModel):
    job_description = str
