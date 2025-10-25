from sqlalchemy import Column, Integer, Text, JSON
from sqlalchemy.dialects.postgresql import JSON
from database import Base

class Product(Base):
    __tablename__ = "Produk"
    id = Column(Integer, primary_key=True, index=True)
    deskripsi = Column("deskripsi", Text, nullable=False)

class ProductTFIDF(Base):
    __tablename__ = "Product_tfidf"
    product_id = Column(Integer, primary_key=True)
    preprocessed_tokens = Column(JSON, nullable=False)
    tfidf_vector = Column(JSON, nullable=False)