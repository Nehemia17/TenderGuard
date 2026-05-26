# HoaxGuard — Implementation Plan
## Pivot dari TenderGuard ke Sistem Deteksi Hoaks

---

## Konsep Project
**HoaxGuard** — Sistem Deteksi Hoaks Berbahasa Indonesia Berbasis NLP & Machine Learning

Sistem ini mengklasifikasikan teks berita/artikel sebagai HOAKS atau VALID menggunakan model ML yang dilatih dengan dataset berita Indonesia. Dilengkapi dashboard interaktif untuk demonstrasi real-time.

---

## Stack Teknologi
- **Training:** Google Colab (scikit-learn, Sastrawi, pandas, nltk)
- **Model:** TF-IDF + Logistic Regression (baseline) → Random Forest / SVM
- **Dashboard:** Streamlit (desain ClickHouse tetap dipertahankan)
- **NLP:** Sastrawi (stemming Bahasa Indonesia), NLTK

---

## Dataset
- **Primer:** Dataset hoaks Indonesia dari Kaggle
  - `fake-news-detection-in-bahasa-indonesia`
  - `indonesian-hoax-news`
- **Sekunder (opsional scraping):** turnbackhoax.id, cekfakta.tempo.co

---

## Struktur Folder Baru
```
GraphGaze-Lab/
├── notebooks/
│   └── HoaxGuard_Training_Colab.py    ← Script untuk Google Colab
├── modules/
│   ├── __init__.py
│   ├── preprocessor.py                ← Text cleaning & NLP pipeline
│   ├── feature_extractor.py           ← TF-IDF, fitur tambahan
│   └── predictor.py                   ← Load model & predict
├── model/
│   └── (diisi setelah training di Colab)
│       ├── hoax_classifier.pkl
│       └── tfidf_vectorizer.pkl
├── data/
│   ├── raw/                           ← Dataset mentah dari Kaggle
│   └── processed/
│       └── dataset_clean.csv
├── dashboard/
│   ├── app.py                         ← Dashboard baru (Streamlit)
│   └── style.css                      ← ClickHouse design (tetap)
├── requirements.txt
├── config.yaml
└── README.md
```

---

## CRISP-DM Framework

### 1. Business Understanding
Hoaks merupakan ancaman nyata terhadap stabilitas sosial dan informasi publik di Indonesia. Sistem ini memberikan alat klasifikasi otomatis bagi masyarakat dan lembaga fact-checker.

### 2. Data Understanding
- Jumlah sampel: Target minimal 5.000 record (hoaks + valid)
- Fitur utama: judul berita, isi artikel, label (0=valid, 1=hoaks)
- Distribusi: Analisis class balance, panjang teks, kata-kata dominan

### 3. Data Preparation
- Case folding (lowercase)
- Remove URL, mention, hashtag, karakter khusus
- Tokenisasi
- Stopword removal (NLTK + kamus stopword Indonesia)
- Stemming (Sastrawi)
- TF-IDF Vectorization

### 4. Modeling
| Model | Keterangan |
|-------|-----------|
| Multinomial Naive Bayes | Baseline |
| Logistic Regression | **Model utama** |
| Random Forest | Ensemble |
| SVM (Linear Kernel) | High-accuracy candidate |

### 5. Evaluation
- Metrik: Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Cross-validation (5-fold)
- ROC-AUC Curve

### 6. Deployment (Nilai Plus ✅)
- Streamlit dashboard real-time
- Input teks bebas → prediksi instan
- Confidence score + kata-kata kunci penyebab

---

## Pembagian Tugas Tim (Template)
| Anggota | Tugas |
|---------|-------|
| Ketua | Business Understanding + Koordinasi |
| Anggota 1 | Data Collection & Data Understanding |
| Anggota 2 | Data Preparation & Preprocessing |
| Anggota 3 | Modeling & Training (Google Colab) |
| Anggota 4 | Evaluation & Dashboard Deployment |

---

## File yang Akan Dibuat
- [x] `modules/preprocessor.py`
- [x] `modules/feature_extractor.py`
- [x] `modules/predictor.py`
- [x] `notebooks/HoaxGuard_Training_Colab.py`
- [x] `dashboard/app.py` (versi baru)
- [x] `requirements.txt` (update)
- [x] `config.yaml` (update)
- [x] `README.md` (update)
- [x] `generate_demo_data.py` (versi baru untuk demo hoaks)
