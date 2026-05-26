"""
generate_demo_data.py
Generator tingkat lanjut untuk HoaxGuard.
Mensimulasikan dataset yang lebih besar dan melatih 3 model (Ensemble/Perbandingan)
beserta perhitungan metrik evaluasi untuk CRISP-DM.
"""
import os
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def generate_large_dummy_data():
    # Kumpulan pola hoaks dan valid
    valid_templates = [
        "Pemerintah resmi menetapkan libur nasional pada hari raya bulan depan.",
        "Bank Indonesia melaporkan pertumbuhan ekonomi kuartal ini naik 5 persen.",
        "Gempa bermagnitudo 5.2 mengguncang wilayah pesisir, tidak berpotensi tsunami.",
        "Kementerian Kesehatan mendistribusikan vaksin polio ke 10 provinsi.",
        "Pertandingan final liga 1 dimenangkan oleh tim tuan rumah dengan skor 2-0.",
        "Harga emas antam hari ini stagnan di angka 1 juta rupiah per gram.",
        "BMKG memprediksi hujan lebat disertai angin kencang di Jabodetabek besok.",
        "Polisi menetapkan tersangka baru dalam kasus korupsi pengadaan infrastruktur.",
        "Presiden menghadiri KTT ASEAN di Jakarta hari ini.",
        "Pendaftaran beasiswa LPDP gelombang kedua resmi dibuka hari ini."
    ]
    
    hoax_templates = [
        "Vaksin covid-19 secara rahasia mengandung chip mikro untuk melacak pergerakan manusia!!!",
        "Kiamat akan terjadi hari Jumat bulan depan, NASA tutupi fakta ini! Sebarkan!",
        "Rebusan daun sirih dan madu diklaim bisa menyembuhkan kanker kronis stadium 4 dalam semalam.",
        "Awas! Jangan makan mi instan dicampur coklat, bisa menyebabkan usus berdarah dan kematian instan!",
        "Alien tertangkap kamera CCTV mendarat di atap gedung DPR, pemerintah diam saja!",
        "Bantuan langsung tunai 50 juta dari pemerintah, klik link ini sekarang sebelum kehabisan!!!",
        "Waspada penculikan anak menggunakan mobil putih, mereka mengambil organ dalam anak-anak!",
        "Terungkap! Bumi sebenarnya datar, satelit hanya tipuan konspirasi elit global.",
        "Minum air es setelah kepanasan bisa membuat pembuluh darah pecah seketika.",
        "Cina kirim jutaan tentara pakai baju sipil untuk kuasai pulau Jawa, ini buktinya!"
    ]
    
    data = []
    # Generate 500 baris dataset sintetik dengan sedikit variasi noise
    np.random.seed(42)
    for i in range(250):
        data.append({"id": f"V{i:03d}", "text": valid_templates[i % 10] + " " + " ".join(np.random.choice(["dan", "di", "ini", "itu"], 2)), "label": 0})
        data.append({"id": f"H{i:03d}", "text": hoax_templates[i % 10] + " " + " ".join(np.random.choice(["!!", "viralkan", "gawat", "awas"], 2)), "label": 1})
        
    return pd.DataFrame(data)

def run():
    print("Membangun model kompleks & dataset simulasi untuk HoaxGuard...")
    
    Path("model").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    df = generate_large_dummy_data()
    df.to_csv("data/processed/dataset_hoax_demo.csv", index=False)
    
    # Preprocessing ringan (simulasi)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]
    
    # Split manual simulasi (80% train, 20% test)
    split_idx = int(0.8 * len(df))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 1. Logistic Regression (Primary)
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # 2. Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    
    # 3. Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Simpan Model
    joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
    joblib.dump(lr_model, "model/hoax_classifier.pkl") # Main model
    joblib.dump(nb_model, "model/nb_classifier.pkl")
    joblib.dump(rf_model, "model/rf_classifier.pkl")
    
    # Hitung Metrik untuk Dashboard (Evaluation Phase CRISP-DM)
    def calc_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "cm": cm
        }
        
    evaluation_results = {
        "LogisticRegression": calc_metrics(y_test, lr_pred),
        "NaiveBayes": calc_metrics(y_test, nb_pred),
        "RandomForest": calc_metrics(y_test, rf_pred)
    }
    
    with open("model/evaluation_metrics.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)
        
    print("✅ Model multi-algoritma & Dataset 500 baris berhasil dibuat!")
    print("🚀 Silakan jalankan: streamlit run dashboard/app.py")

if __name__ == "__main__":
    run()
