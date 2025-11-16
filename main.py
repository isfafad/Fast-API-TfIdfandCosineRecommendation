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
import numpy as np

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
    Dipanggil oleh Echo saat ada perubahan produk (tambah/ubah/hapus).
    """
    # 1. Ambil semua produk
    products = db.query(Product).all()
    if not products:
        # Opsional: tetap izinkan sync kosong?
        db.query(ProductTFIDF).delete()
        db.commit()
        return {"message": "No products found. Cleared TF-IDF table."}

    # 2. Preprocess semua deskripsi
    docs = []
    product_data = []  # list of (product_id, tokens)
    for p in products:
        tokens = preprocess_terarah(p.deskripsi)  # pastikan ini mengembalikan List[str]
        docs.append(tokens)
        product_data.append((p.id, tokens))

    # 3. Hitung IDF global berdasarkan semua dokumen
    idf = compute_idf(docs)

    # 4. Hapus semua entri lama (transaksional)
    db.query(ProductTFIDF).delete()

    # 5. Hitung TF-IDF per produk dan simpan
    for product_id, tokens in product_data:
        tf = compute_tf(tokens)
        tfidf_vec = compute_tfidf(tf, idf)
        tfidf_entry = ProductTFIDF(
            product_id=product_id,
            preprocessed_tokens=tokens,      # List[str] → pastikan kolom di DB bisa menyimpan ini (misal JSON)
            tfidf_vector=tfidf_vec           # Dict[str, float] → juga disimpan sebagai JSON
        )
        db.add(tfidf_entry)

    # 6. Commit sekali saja
    db.commit()

    return {"message": f"TF-IDF successfully synced for {len(products)} products"}

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

    # Ambil semua vektor lain dengan JOIN ke tabel Product
    all_vectors = (
        db.query(ProductTFIDF, Product)
        .join(Product, ProductTFIDF.product_id == Product.id)
        .filter(ProductTFIDF.product_id != product_id)
        .all()
    )

    similarities = []
    for tfidf_data, product_data in all_vectors:
        sim = cosine_similarity(target_vec, tfidf_data.tfidf_vector)
        if sim > 0:
            similarities.append({
                "product_id": product_data.id,
                "nama": product_data.nama,
                "gambar": product_data.gambar,
                "stok": product_data.stok,
                "harga": int(product_data.harga),  # Convert ke int
                "similarity": round(sim, 4)
            })

    # Urutkan dan ambil top_k
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top = similarities[:top_k]

    return {
        "target_product_id": product_id,
        "recommendations": top
    }