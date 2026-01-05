from pydantic import BaseModel

#Schéma d'entrée
class ProfileRequest(BaseModel):
    text: str

# Schéma de sortie (optionnel mais pro)
class ProfileResponse(BaseModel):
    summary: list
    skills: list
    profile_type: str
    level: str
    axes: dict
    score: int