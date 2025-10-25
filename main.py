from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from database import get_db
from database import SessionLocal, engine
from models import Product, ProductTFIDF
from recommender import (
    preprocess_terarah,
    compute_tf,
    compute_idf,
    compute_tfidf,
    cosine_similarity
)
from typing import List, Dict
import logging

# Buat tabel jika belum ada
from models import Base

app = FastAPI(title="Recommender Microservice")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/sync-tfidf")
def sync_tfidf(db: Session = Depends(get_db)):
    """
    Hitung ulang TF-IDF untuk SEMUA produk dan simpan ke product_tfidf.
    Dipanggil oleh Echo saat ada perubahan produk.
    """
    # Ambil semua produk
    products = db.query(Product).all()
    if not products:
        raise HTTPException(status_code=404, detail="No products found")

    docs = []
    product_dict = {}
    for p in products:
        tokens = preprocess_terarah(p.deskripsi)
        docs.append(tokens)
        product_dict[p.id] = tokens

    # Hitung IDF global
    idf = compute_idf(docs)

    # Simpan ke DB
    db.query(ProductTFIDF).delete()  # Hapus semua dulu (karena IDF berubah)
    for p in products:
        tokens = product_dict[p.id]
        if not tokens:
            tfidf_vec = {}
        else:
            tf = compute_tf(tokens)
            tfidf_vec = compute_tfidf(tf, idf)
        tfidf_entry = ProductTFIDF(
            product_id=p.id,
            preprocessed_tokens=tokens,
            tfidf_vector=tfidf_vec
        )
        db.add(tfidf_entry)
    db.commit()
    return {"message": f"TF-IDF synced for {len(products)} products"}

@app.get("/recommend/{product_id}")
def recommend(product_id: int, top_k: int = 4, db: Session = Depends(get_db)):
    """
    Berikan rekomendasi berdasarkan cosine similarity.
    """
    # Ambil target
    target = db.query(ProductTFIDF).filter(ProductTFIDF.product_id == product_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Product TF-IDF not found. Run /sync-tfidf first.")

    target_vec = target.tfidf_vector

    # Ambil semua vektor lain
    all_vectors = db.query(ProductTFIDF).filter(ProductTFIDF.product_id != product_id).all()

    similarities = []
    for other in all_vectors:
        sim = cosine_similarity(target_vec, other.tfidf_vector)
        if sim > 0:
            similarities.append((other.product_id, sim))

    # Urutkan dan ambil top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top = similarities[:top_k]

    return {
        "target_product_id": product_id,
        "recommendations": [
            {"product_id": pid, "similarity": round(score, 4)} for pid, score in top
        ]
    }