# Technical Requirements Document (TRD)
**Project Name:** HoaxGuard  
**Version:** 1.0  
**Target:** GEMASTIK XVIII 2026 (Kategori Penambangan Data)  

## 1. Arsitektur Sistem
HoaxGuard dirancang dengan pendekatan modular berbasis micro-architecture.
1. **Data Layer:** File CSV/dummy data berisi teks bahasa Indonesia.
2. **Preprocessing Layer:** Modul NLP kustom menggunakan Python, RegEx, dan Sastrawi.
3. **Modeling Layer:** Scikit-Learn (TF-IDF Vectorizer + ML Classifier).
4. **Presentation Layer:** Aplikasi web interaktif berbasis Streamlit.

## 2. Stack Teknologi
- **Bahasa Pemrograman:** Python 3.10+
- **Machine Learning:** `scikit-learn` (Logistic Regression / Naive Bayes / RandomForest)
- **Natural Language Processing (NLP):** `Sastrawi` (Stemming Indonesia), `nltk`, Regex (`re`)
- **Web Dashboard:** `streamlit`
- **Data Manipulation:** `pandas`, `numpy`
- **Serialization:** `joblib` (untuk eksport/import `.pkl` model AI)

## 3. Alur Pemrosesan Teks (NLP Pipeline)
Setiap teks yang masuk (baik saat *training* maupun *inference*) harus melalui pipa transformasi berikut di dalam `modules/preprocessor.py`:
1. **Unicode Normalization & Lowercasing:** Menghilangkan anomali *encoding* dan menyamakan huruf.
2. **Regex Cleaning:** Menghapus tag HTML, URL (`http://`), *mentions* (`@`), dan *hashtags* (`#`).
3. **Punctuation & Number Removal:** Kecuali untuk ekstraksi fitur meta, angka dan tanda baca dihilangkan.
4. **Stopword Removal:** Menghapus kata hubung (dan, di, ke, yang) menggunakan NLTK/Sastrawi.
5. **Stemming (Sastrawi):** Mengembalikan kata berimbuhan ke kata dasar (misal: "menyebarkan" -> "sebar").

## 4. Ekstraksi Fitur (Feature Engineering)
Teks yang sudah bersih (Tokens) akan diproses oleh `modules/feature_extractor.py`:
- **TF-IDF Vectorizer:** Mengukur seberapa unik dan penting sebuah kata dalam suatu dokumen terhadap seluruh corpus.
- **Statistical Meta-features (Opsional):** Ekstraksi jumlah huruf kapital dan jumlah tanda baca provokatif (seperti `!`, `?`) yang sangat sering digunakan dalam *broadcast* hoaks.

## 5. Metodologi Model Machine Learning
- Menggunakan **Supervised Machine Learning**.
- **Model Utama:** Logistic Regression (dipilih karena ringan, interpretabel, dan bekerja sangat baik dengan dimensi tinggi seperti matriks kata TF-IDF).
- Model harus mengembalikan *Probability Distribution* (`predict_proba`) untuk memunculkan fitur *Confidence Score* di UI.

## 6. Persyaratan Penyebaran (Deployment)
- Aplikasi harus dapat dijalankan secara lokal dengan mulus.
- Perintah eksekusi bersifat *zero-config*: `streamlit run dashboard/app.py`.
- Seluruh *styling* harus dikendalikan secara mandiri oleh `style.css` agar mem-bypass desain *default* framework web.
