# Product Requirements Document (PRD)
**Project Name:** HoaxGuard  
**Version:** 1.0  
**Target:** GEMASTIK XVIII 2026 (Kategori Penambangan Data)  

## 1. Visi Produk
Membangun platform berbasis kecerdasan buatan (AI) yang mampu menganalisis, memverifikasi, dan mendeteksi berita atau pesan berantai hoaks berbahasa Indonesia secara *real-time* untuk membantu menekan penyebaran disinformasi (hoaks) demi kemandirian bangsa.

## 2. Latar Belakang (Mengapa project ini ada?)
Hoaks menyebar sangat cepat melalui grup pesan singkat (seperti WhatsApp) dan media sosial. Masyarakat umum dan lembaga *fact-checker* (pengecek fakta) membutuhkan alat bantu otomatis yang dapat memindai pola bahasa provokatif dan mengidentifikasi klaim palsu tanpa harus menunggu verifikasi manual dari jurnalis. Project ini dibuat untuk memenuhi kriteria tema GEMASTIK 2026: **Deteksi Hoaks**.

## 3. Target Pengguna
1. **Masyarakat Umum:** Untuk memvalidasi pesan berantai di grup WA/keluarga sebelum disebarluaskan.
2. **Fact-Checker & Jurnalis:** Sebagai alat pemindaian awal (*triage*) untuk artikel-artikel yang memiliki pola linguistik mencurigakan.

## 4. Fitur Utama (Core Features)
### 4.1. Real-time Hoax Analyzer
- Pengguna dapat menyalin-tempel (*copy-paste*) teks panjang ke dalam area analisis.
- Sistem akan memunculkan status klasifikasi yang sangat jelas: **🚨 HOAKS** atau **✅ VALID**.
- Menampilkan persentase metrik kepercayaan sistem (*Confidence Score*).

### 4.2. Explainable AI (XAI) Panel
- Tidak sekadar memvonis "Hoaks", sistem harus menjelaskan *mengapa*.
- Fitur ini membongkar proses NLP: menampilkan teks yang sudah di-*stemming* dan menyorot (Highlight) kata-kata atau *red flags* (seperti tanda seru berlebihan atau kata provokatif) yang menyebabkan sistem mendeteksinya sebagai hoaks.

### 4.3. Dataset & Reference Viewer
- Modul bagi juri kompetisi untuk melihat distribusi statistik dataset yang kita gunakan untuk melatih AI (menampilkan rasio berita asli vs hoaks).

## 5. Kriteria Penerimaan (Success Metrics)
- Model harus dapat merespons analisis teks di bawah 2 detik.
- Aplikasi memiliki *user interface* (UI) yang terlihat profesional dan elegan.
- Model sanggup memproses *slang* atau format pesan berantai khas Indonesia (dengan bantuan Sastrawi & regex rules).
