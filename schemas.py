from pydantic import BaseModel
from typing import List

class RecommendationItem(BaseModel):
    product_id: int
    nama: str
    gambar: str
    stok: int
    harga: float
    similarity: float

class RecommendationResponse(BaseModel):
    target_product_id: int
    recommendations: List[RecommendationItem]