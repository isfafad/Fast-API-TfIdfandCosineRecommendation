import re
import math
from collections import Counter
from typing import List, Dict, Tuple

stopwords = set([
    'yang', 'dan', 'untuk', 'dari', 'pada', 'dengan', 'oleh', 'atau', 'juga',
    'dalam', 'ke', 'di', 'ini', 'itu', 'sebagai', 'adalah', 'dapat', 'digunakan',
    'cocok', 'guna', 'tas', 'kulit', 'warna', 'berwarna', 'asli'
])

def preprocess_terarah(deskripsi: str) -> List[str]:
    deskripsi = deskripsi.lower()
    fitur = []

    bahan_match = re.search(r'(?:kulit|berbahan)\s+([a-z\s]+?)(?:,|\.)', deskripsi)
    if bahan_match:
        fitur.append(bahan_match.group(1))

    warna_match = re.search(r'(?:warna|berwarna)\s+([a-z\s]+?)(?:,|\.)', deskripsi)
    if warna_match:
        fitur.append(warna_match.group(1))

    fungsi_match = re.search(r'(?:cocok digunakan untuk|digunakan untuk|ideal untuk)\s+([a-z\s]+?)(?:,|\.)', deskripsi)
    if fungsi_match:
        fitur.append(fungsi_match.group(1))

    teks = ' '.join(fitur)
    tokens = re.findall(r'\b\w+\b', teks)
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

def compute_tf(doc: List[str]) -> Dict[str, float]:
    if not doc:
        return {}
    count = Counter(doc)
    total = len(doc)
    return {term: freq / total for term, freq in count.items()}


def compute_idf(all_docs: List[List[str]]) -> Dict[str, float]:
    N = len(all_docs)
    if N == 0:
        return {}
    all_terms = set(term for doc in all_docs for term in doc)
    idf = {}
    for term in all_terms:
        doc_count = sum(1 for doc in all_docs if term in doc)
        # Hindari log(0) — tapi doc_count minimal 1 karena term berasal dari dokumen
        idf[term] = math.log(N / doc_count)
    return idf

def compute_tfidf(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
    return {term: tf[term] * idf.get(term, 0.0) for term in tf}


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    terms = set(vec1.keys()) | set(vec2.keys())
    dot = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in terms)
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)