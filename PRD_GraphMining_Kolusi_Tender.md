# Product Requirements Document (PRD)
## Graph Mining untuk Deteksi Kolusi Tender di Data Pengadaan LPSE Nasional

---

**Versi:** 1.0  
**Tanggal:** Mei 2026  
**Status:** Draft  
**Kompetisi Target:** GEMASTIK — Kategori Pelayanan Publik

---

## 1. Latar Belakang & Konteks

### 1.1 Permasalahan

KPK mencatat bahwa 57% kasus korupsi di Indonesia terjadi di sektor konstruksi dan pengadaan barang/jasa pemerintah. Salah satu modus yang paling sulit dideteksi adalah **bid-rigging** — praktik kolusi antar vendor di mana sekelompok perusahaan bergantian "menang" tender secara terkoordinasi, sementara peserta lain hanya berpura-pura berkompetisi.

Tantangan utama deteksi bid-rigging saat ini:

- Volume data sangat besar: ratusan ribu tender per tahun di seluruh Indonesia
- Pendekatan yang ada (ICW, Opentender) hanya menggunakan indikator statistik sederhana — belum memodelkan **relasi jaringan** antar perusahaan
- Tidak ada sistem otomatis berbasis Graph ML yang tersedia secara publik di Indonesia
- Proses audit manual tidak skalabel dan rawan human error

### 1.2 Peluang

Data pengadaan pemerintah Indonesia tersedia secara publik melalui LPSE dan LKPP. Dengan memodelkan relasi keikutsertaan vendor sebagai graf, pola kolusi yang tidak terlihat secara statistik dapat terdeteksi melalui analisis struktur jaringan dan machine learning.

### 1.3 Urgensi

Tidak ada sistem serupa yang dipublikasikan di Indonesia dengan pendekatan Graph ML. Paper terbaru yang ditemukan (ResearchGate 2016) masih menggunakan metodologi konvensional. Ini merupakan **gap riset dan produk** yang nyata.

---

## 2. Tujuan Produk

### 2.1 Tujuan Utama

Membangun sistem deteksi kolusi tender berbasis graph mining yang mampu secara otomatis mengidentifikasi kluster vendor mencurigakan dan memberikan risk score per tender dari data pengadaan LPSE nasional.

### 2.2 Sasaran Terukur

| Sasaran | Metrik Keberhasilan |
|---|---|
| Deteksi pola bid-rigging | Precision ≥ 70% pada data berlabel |
| Cakupan data | ≥ 1.000 tender dari 1 provinsi/kabupaten |
| Waktu proses | Pipeline end-to-end selesai < 30 menit |
| Visualisasi | Peta jaringan vendor interaktif dapat diakses |
| Risk score | Setiap tender memiliki skor risiko 0–1 dengan penjelasan |

---

## 3. Pengguna Target (Stakeholder)

### 3.1 Pengguna Primer

| Persona | Deskripsi | Kebutuhan Utama |
|---|---|---|
| **Analis KPK / BPKP** | Auditor pemerintah yang menginvestigasi pengadaan | Prioritisasi tender untuk diinvestigasi |
| **Jurnalis Investigasi** | Wartawan dari ICW, Tempo, dll | Temuan berbasis data yang dapat dipublikasikan |
| **Peneliti Kebijakan** | Akademisi / lembaga riset antikorupsi | Dataset dan metodologi yang dapat direplikasi |

### 3.2 Pengguna Sekunder

- Tim pengadaan internal instansi pemerintah (untuk audit mandiri)
- Mahasiswa / peneliti yang ingin replikasi studi

---

## 4. Fitur Produk

### 4.1 Modul 1 — Data Ingestion & Scraping

**Deskripsi:** Pipeline otomatis untuk mengumpulkan dan membersihkan data tender dari sumber publik.

**Fitur:**
- Scraper data tender dari opentender.net (per kabupaten/kementerian)
- Integrasi dengan API resmi LKPP (`lpse.lkpp.go.id/api`)
- Opsional: pengambilan data historis dari inaproc.id
- Normalisasi nama perusahaan (deduplication fuzzy matching)
- Penyimpanan ke format tabular standar (CSV / SQLite)

**Input:** Parameter wilayah / instansi pemerintah target  
**Output:** Dataset tender bersih dengan kolom: `tender_id`, `vendor_id`, `nama_vendor`, `nilai_penawaran`, `status (menang/kalah)`, `tanggal`, `instansi`

---

### 4.2 Modul 2 — Graf Construction

**Deskripsi:** Pembangunan graf dari data tender untuk merepresentasikan relasi antar vendor.

**Definisi Graf:**
- **Node:** Perusahaan/vendor
- **Edge:** Dua vendor pernah mengikuti tender yang sama
- **Edge weight:** Frekuensi co-participation (semakin sering ikut tender bersama, bobot semakin tinggi)
- **Node attribute:** Win rate, jumlah tender diikuti, rata-rata selisih harga penawaran dari pemenang

**Fitur:**
- Konstruksi *co-participation graph* dari data tender
- Konstruksi *win-loss relationship graph* (siapa yang kalah saat vendor X menang)
- Perhitungan fitur graf: degree centrality, clustering coefficient, betweenness centrality
- Ekspor graf ke format NetworkX / PyTorch Geometric

---

### 4.3 Modul 3 — Deteksi Anomali & Kolusi

**Deskripsi:** Penerapan algoritma ML untuk mendeteksi pola abnormal dalam graf vendor.

**Metode Deteksi:**

| Metode | Kegunaan |
|---|---|
| **Isolation Forest** | Deteksi node/subgraf anomali secara statistik |
| **Louvain Community Detection** | Identifikasi kluster vendor yang terlalu kohesif |
| **Graph Neural Network (GNN)** | Klasifikasi node mencurigakan berdasarkan struktur lokal |
| **Indikator heuristik** | Win-rate tidak wajar, selisih penawaran terlalu kecil/konsisten |

**Sinyal Kolusi yang Dideteksi:**
- Vendor A selalu kalah ketika Vendor B mengikuti tender yang sama (cover bidding)
- Kluster perusahaan dengan pola rotasi kemenangan (vendor bergantian menang)
- Penawaran harga yang terlalu mirip antar peserta (bid suppression)
- Hubungan kepemilikan/alamat yang tersembunyi di balik nama perusahaan berbeda

---

### 4.4 Modul 4 — Risk Scoring

**Deskripsi:** Setiap tender diberi skor risiko kolusi berdasarkan gabungan sinyal dari Modul 3.

**Risk Score Formula:**
- Skor 0.0–1.0 (semakin tinggi = semakin mencurigakan)
- Komponen: structural anomaly score + heuristic indicator score + historical pattern score
- Threshold otomatis: Low (< 0.4), Medium (0.4–0.7), High (> 0.7)

**Output per Tender:**
- Risk score dan kategori risiko
- Daftar sinyal spesifik yang memicu skor tinggi
- Vendor-vendor yang terlibat dalam jaringan mencurigakan

---

### 4.5 Modul 5 — Visualisasi & Dashboard

**Deskripsi:** Antarmuka visual untuk eksplorasi hasil analisis.

**Fitur:**
- Peta jaringan vendor interaktif (berbasis Pyvis / D3.js / Gephi export)
- Tabel ranking tender berdasarkan risk score
- Filter by: instansi pemerintah, rentang tanggal, ambang risk score
- Highlight kluster vendor mencurigakan dengan warna berbeda
- Export laporan ke PDF/CSV

---

## 5. Ruang Lingkup (Scope)

### 5.1 In Scope (MVP)

- 1 provinsi atau kabupaten sebagai pilot (direkomendasikan: DKI Jakarta atau Jawa Barat)
- Data minimal 1 tahun terakhir
- Fokus pada sektor konstruksi (paling rawan berdasarkan data KPK)
- Dashboard sederhana berbasis web (Streamlit / Flask)

### 5.2 Out of Scope (MVP)

- Pemrosesan seluruh data nasional secara real-time
- Integrasi langsung dengan sistem KPK/BPKP
- Analisis korupsi di luar sektor pengadaan barang/jasa

---

## 6. Sumber Data

| Sumber | URL | Tipe | Ketersediaan |
|---|---|---|---|
| Opentender | opentender.net | Web scraping / CSV download | Publik, gratis |
| API LKPP | lpse.lkpp.go.id/api | REST API | Publik, gratis |
| InaProc | inaproc.id | Web scraping | Publik, gratis |

---

## 7. Asumsi & Risiko

### 7.1 Asumsi

- Data publik LPSE cukup bersih untuk analisis (nama vendor konsisten antar tender)
- Tidak diperlukan izin khusus untuk mengakses dan menganalisis data publik
- Tim memiliki atau akan mempelajari Python, NetworkX, dan dasar graph ML

### 7.2 Risiko

| Risiko | Probabilitas | Dampak | Mitigasi |
|---|---|---|---|
| Data tidak konsisten (nama perusahaan berbeda-beda) | Tinggi | Tinggi | Fuzzy matching + normalisasi manual |
| API LKPP tidak stabil / rate-limited | Sedang | Sedang | Caching lokal, fallback ke opentender.net |
| False positive tinggi pada deteksi | Sedang | Tinggi | Kombinasikan beberapa sinyal, tambahkan threshold manual |
| Kurva belajar GNN terlalu curam | Sedang | Sedang | Mulai dengan Isolation Forest, GNN sebagai enhancement |

---

## 8. Kriteria Keberhasilan (GEMASTIK)

- Sistem dapat berjalan end-to-end pada data nyata (bukan dummy)
- Minimal 1 kasus kolusi yang dapat dijelaskan secara naratif dari hasil sistem
- Visualisasi jaringan yang jelas dan dapat dipresentasikan
- Metodologi dapat direplikasi dan didokumentasikan

---

## 9. Timeline (Estimasi)

| Fase | Durasi | Deliverable |
|---|---|---|
| Fase 1: Data & Graph | 2 minggu | Pipeline scraping + konstruksi graf |
| Fase 2: Deteksi | 2 minggu | Model Isolation Forest + GNN baseline |
| Fase 3: Dashboard | 1 minggu | Web dashboard + visualisasi |
| Fase 4: Validasi & Demo | 1 minggu | Laporan hasil + persiapan presentasi |

**Total: ~6 minggu**

---

*Dokumen ini adalah living document dan akan diperbarui seiring iterasi pengembangan.*
