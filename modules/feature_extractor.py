"""
feature_extractor.py — TF-IDF vectorization + fitur tambahan.
Digunakan baik saat training (Colab) maupun inferensi (dashboard).
"""
import numpy as np
import joblib
from pathlib import Path
from typing import List, Optional, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def build_tfidf_vectorizer(
    max_features: int = 50_000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Buat TF-IDF vectorizer dengan parameter yang sudah disetel."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        analyzer="word",
        strip_accents="unicode",
    )


def extract_extra_features(texts: List[str]) -> np.ndarray:
    """
    Ekstraksi fitur tambahan berbasis statistik teks (bukan TF-IDF):
    - panjang teks (jumlah karakter)
    - jumlah kata
    - jumlah tanda seru
    - jumlah huruf kapital
    - jumlah tanda tanya
    """
    feats = []
    for t in texts:
        feats.append([
            len(t),
            len(t.split()),
            t.count("!"),
            sum(1 for c in t if c.isupper()),
            t.count("?"),
        ])
    return np.array(feats, dtype=np.float32)


def save_vectorizer(vectorizer: TfidfVectorizer, path: Optional[Path] = None) -> Path:
    path = path or (MODEL_DIR / "tfidf_vectorizer.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved → {path}")
    return path


def load_vectorizer(path: Optional[Path] = None) -> TfidfVectorizer:
    path = path or (MODEL_DIR / "tfidf_vectorizer.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer tidak ditemukan: {path}")
    return joblib.load(path)


def transform_texts(
    texts: List[str],
    vectorizer: TfidfVectorizer,
    include_extra: bool = True,
) -> np.ndarray:
    """Transform list teks menjadi feature matrix untuk prediksi."""
    from scipy.sparse import hstack, csr_matrix
    tfidf_matrix = vectorizer.transform(texts)
    if include_extra:
        extra = csr_matrix(extract_extra_features(texts))
        return hstack([tfidf_matrix, extra])
    return tfidf_matrix


if __name__ == "__main__":
    sample_texts = [
        "vaksin mengandung chip mikro pemerintah bohong",
        "presiden meresmikan jembatan baru di kalimantan",
    ]
    vec = build_tfidf_vectorizer()
    vec.fit(sample_texts)
    result = transform_texts(sample_texts, vec, include_extra=True)
    print("Feature matrix shape:", result.shape)
