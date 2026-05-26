"""
predictor.py — Memuat model dan vectorizer yang telah dilatih (misalnya dari Colab)
dan melakukan prediksi terhadap input teks baru.
"""
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union

from .preprocessor import clean_text
from .feature_extractor import load_vectorizer, transform_texts

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def load_classifier(path: Union[str, Path] = None):
    """Load model klasifikasi (Logistic Regression / RandomForest)."""
    path = path or (MODEL_DIR / "hoax_classifier.pkl")
    if not Path(path).exists():
        raise FileNotFoundError(f"Model classifier tidak ditemukan: {path}")
    return joblib.load(path)


def get_feature_importances(classifier, vectorizer, top_n=20) -> list:
    """Mengambil fitur (kata) yang paling berkontribusi untuk kelas HOAKS."""
    if not hasattr(classifier, "coef_"):
        return []
    
    # Ambil bobot untuk kelas Hoaks (asumsi kelas 1 = Hoaks)
    coef = classifier.coef_[0]
    # Gabungkan nama fitur dari TFIDF
    vocab = vectorizer.get_feature_names_out()
    
    # Karena ada fitur ekstra, vocab TFIDF mungkin lebih pendek dari coef
    n_tfidf = len(vocab)
    
    coef_tfidf = coef[:n_tfidf]
    
    # Urutkan berdasarkan bobot terbesar (paling mendukung kelas 1)
    top_indices = np.argsort(coef_tfidf)[::-1][:top_n]
    
    top_features = [(vocab[i], float(coef_tfidf[i])) for i in top_indices]
    return top_features


def predict_text(text: str, vectorizer, classifier, include_extra: bool = True) -> Dict[str, Any]:
    """
    1. Preprocess teks
    2. Ekstraksi fitur
    3. Prediksi dan ambil confidence score
    """
    # 1. Cleaning
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "prediction": 0,
            "label": "VALID",
            "confidence": 0.0,
            "cleaned_text": "",
        }
        
    # 2. Ekstraksi
    # Ekstra fitur statistik diambil dari teks mentah, tfidf dari cleaned
    feats = transform_texts([cleaned], vectorizer, include_extra=False)
    
    if include_extra:
        from .feature_extractor import extract_extra_features
        from scipy.sparse import csr_matrix, hstack
        extra = csr_matrix(extract_extra_features([text]))
        feats = hstack([feats, extra])
        
    # 3. Prediksi
    pred = classifier.predict(feats)[0]
    
    # Cek apakah classifier memiliki atribut predict_proba
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(feats)[0]
        conf = float(probs[pred])
    else:
        conf = 1.0
        
    label = "HOAKS" if pred == 1 else "VALID"
    
    return {
        "prediction": int(pred),
        "label": label,
        "confidence": conf,
        "cleaned_text": cleaned,
    }

if __name__ == "__main__":
    pass
