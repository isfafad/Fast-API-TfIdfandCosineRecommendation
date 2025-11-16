from sqlalchemy import Column, Integer, Text, JSON, String, Float
from sqlalchemy.dialects.postgresql import JSON
from database import Base

class Product(Base):
    __tablename__ = "Produk"
    id = Column(Integer, primary_key=True, index=True)
    nama = Column("nama", String, nullable=False)
    deskripsi = Column("deskripsi", Text, nullable=False)
    stok = Column("stok", Integer, nullable=False)
    harga = Column("harga", Float, nullable=False)
    berat_gram = Column("berat_gram", Float, nullable=False)
    gambar = Column("gambar", String, nullable=False)

class ProductTFIDF(Base):
    __tablename__ = "Product_tfidf"
    product_id = Column(Integer, primary_key=True)
    preprocessed_tokens = Column(JSON, nullable=False)
    tfidf_vector = Column(JSON, nullable=False)