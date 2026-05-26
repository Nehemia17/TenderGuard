"""
HoaxGuard_Training_Colab.py

Skrip ini dirancang untuk dijalankan di Google Colab.
Salin isi skrip ini ke dalam cell Google Colab atau buka file ini di sana.

Persiapan di Colab:
!pip install Sastrawi pandas scikit-learn nltk
"""

import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================
# 1. INISIALISASI NLP & PREPROCESSING
# ==========================================
print("Inisialisasi Sastrawi...")
factory = StemmerFactory()
stemmer = factory.create_stemmer()
sw_factory = StopWordRemoverFactory()
sw_remover = sw_factory.create_stop_word_remover()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Hapus URL
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Hapus Karakter Khusus & Tanda Baca
    text = re.sub(r'[^\w\s]', ' ', text)
    # Hapus Angka
    text = re.sub(r'\d+', ' ', text)
    # Stopword Removal
    text = sw_remover.remove(text)
    # Stemming
    text = stemmer.stem(text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 2. DATASET LOADING (SIMULASI/DUMMY)
# ==========================================
# Di Colab asli, Anda bisa me-load data dari CSV Kaggle:
# df = pd.read_csv("indonesian_hoax_dataset.csv")

print("Membuat Dataset Sintetis untuk Simulasi Training...")
# 0 = Valid, 1 = Hoaks
dummy_data = [
    ("Pemerintah segera meresmikan jalan tol baru sepanjang 50 km", 0),
    ("Presiden mengadakan kunjungan kenegaraan ke Jepang", 0),
    ("Vaksin covid-19 mengandung microchip untuk melacak manusia!!", 1),
    ("Kiamat akan terjadi hari Jumat bulan depan, sebarkan jika peduli!", 1),
    ("Bawang putih ampuh sembuhkan segala penyakit kronis dalam semalam", 1),
    ("BMKG memperkirakan cuaca cerah di wilayah Jabodetabek besok", 0),
    ("Alien terekam kamera mendarat di monas tadi malam, ini videonya!", 1),
    ("Inflasi tahun ini turun 2 persen berkat kebijakan ekonomi makro", 0),
    ("Jangan minum es setelah makan bakso karena bisa bikin kanker lambung seketika", 1),
    ("Polda Metro Jaya menangkap komplotan pencuri motor di Jakarta Selatan", 0)
]

# Perbanyak dataset dengan variasi untuk mensimulasikan training
df_raw = pd.DataFrame(dummy_data * 50, columns=["text", "label"])

print(f"Dataset dimuat dengan jumlah baris: {len(df_raw)}")

# ==========================================
# 3. PREPROCESSING PIPELINE
# ==========================================
print("Memulai Preprocessing Teks (bisa memakan waktu)...")
df_raw['clean_text'] = df_raw['text'].apply(clean_text)

# Hapus baris kosong
df_raw = df_raw[df_raw['clean_text'].str.len() > 0]

X = df_raw['clean_text']
y = df_raw['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. EKSTRAKSI FITUR (TF-IDF)
# ==========================================
print("Proses TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================================
# 5. MODELING & TRAINING
# ==========================================
print("Training Model (Logistic Regression)...")
model = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)
model.fit(X_train_vec, y_train)

# ==========================================
# 6. EVALUATION
# ==========================================
print("Evaluasi Model...")
y_pred = model.predict(X_test_vec)

print("\n--- Akurasi ---")
print(accuracy_score(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["VALID", "HOAKS"]))

# ==========================================
# 7. EXPORT MODEL
# ==========================================
print("Menyimpan Model dan Vectorizer...")
Path("model").mkdir(exist_ok=True)
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
joblib.dump(model, "model/hoax_classifier.pkl")

print("Training Selesai! Model siap di-download dan dipindahkan ke folder 'model/' di project Anda.")
