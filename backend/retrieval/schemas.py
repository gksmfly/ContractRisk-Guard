# backend/retrieval/schemas.py
from pydantic import BaseModel


class LegalBasis(BaseModel):
    law: str
    article: str
    description: str
