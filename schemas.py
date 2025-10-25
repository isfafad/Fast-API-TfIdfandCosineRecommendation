from pydantic import BaseModel
from typing import List

class RecommendationItem(BaseModel):
    product_id: int
    similarity: float

class RecommendationResponse(BaseModel):
    target_product_id: int
    recommendations: List[RecommendationItem]