"""
Modul 1 — Data Ingestion & Scraping
====================================
Pipeline otomatis untuk mengumpulkan dan membersihkan data tender
dari sumber publik: opentender.net, LKPP API, dan inaproc.id.

Sesuai TRD Section 2.1
"""

import os
import json
import time
import logging
import argparse
import sqlite3
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
import yaml
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process
from tqdm import tqdm

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tenderguard.scraper")

# ─────────────────────────────────────────────
# Konfigurasi endpoint
# ─────────────────────────────────────────────
OPENTENDER_BASE_URL = "https://opentender.net/api/search"
LKPP_BASE_URL       = "https://lpse.lkpp.go.id/eproc4/dt/lelang"
INAPROC_BASE_URL    = "https://inaproc.id/api/contract"

FIELDS_REQUIRED = [
    "id",
    "title",
    "buyers.name",
    "lots.bids.bidders.name",
    "lots.bids.isWinning",
    "lots.bids.price.netAmount",
    "date",
]

# ─────────────────────────────────────────────
# Normalisasi nama vendor (TRD 2.1.3)
# ─────────────────────────────────────────────

def normalize_vendor_name(name: str) -> str:
    """Normalisasi dasar nama perusahaan."""
    name = str(name).upper().strip()
    prefixes = [
        "PT.", "PT ", "CV.", "CV ", "UD.", "UD ",
        "KOPERASI ", "FIRMA ", "FA.", "FA ", "TBK",
    ]
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):].strip()
    # Hapus tanda baca berlebih
    name = name.strip(".,- ")
    return name


def deduplicate_vendors(names: list[str], threshold: int = 85) -> dict:
    """
    Kelompokkan nama vendor yang mirip menggunakan fuzzy matching.
    threshold: skor kemiripan minimum (0–100) untuk dianggap sama.
    Mengembalikan dict {nama_asli: nama_canonical}.
    """
    canonical_map: dict[str, str] = {}
    canonical_list: list[str] = []

    for name in names:
        norm = normalize_vendor_name(name)
        if not norm:
            canonical_map[name] = name
            continue
        match = process.extractOne(norm, canonical_list, scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            canonical_map[name] = match[0]
        else:
            canonical_list.append(norm)
            canonical_map[name] = norm

    return canonical_map


# ─────────────────────────────────────────────
# Schema SQLite (TRD 2.1.2)
# ─────────────────────────────────────────────

CREATE_TENDER_TABLE = """
CREATE TABLE IF NOT EXISTS tender (
    tender_id       TEXT PRIMARY KEY,
    judul           TEXT,
    instansi        TEXT,
    tanggal         DATE,
    nilai_hps       REAL,
    sumber          TEXT
);
"""

CREATE_PESERTA_TABLE = """
CREATE TABLE IF NOT EXISTS peserta (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tender_id       TEXT REFERENCES tender(tender_id),
    vendor_id       TEXT,
    nama_asli       TEXT,
    nilai_penawaran REAL,
    is_winner       BOOLEAN,
    rank            INTEGER
);
"""

CREATE_VENDOR_TABLE = """
CREATE TABLE IF NOT EXISTS vendor (
    vendor_id       TEXT PRIMARY KEY,
    nama_canonical  TEXT,
    nama_aliases    TEXT,
    alamat          TEXT,
    npwp            TEXT
);
"""


def init_database(db_path: str = "data/tender.db") -> sqlite3.Connection:
    """Inisialisasi database SQLite dengan skema yang diperlukan."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(CREATE_TENDER_TABLE)
    cur.execute(CREATE_PESERTA_TABLE)
    cur.execute(CREATE_VENDOR_TABLE)
    conn.commit()
    logger.info(f"Database diinisialisasi: {db_path}")
    return conn


# ─────────────────────────────────────────────
# Scraper — Opentender.net (TRD 2.1.1 Target 1)
# ─────────────────────────────────────────────

class OpentenderScraper:
    """Scraper untuk opentender.net menggunakan REST API publik."""

    BASE_URL = OPENTENDER_BASE_URL

    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TenderGuard/1.0 (Research — GEMASTIK)",
            "Accept": "application/json",
        })
        self.delay = config.get("rate_limit_delay", 0.1)

    def fetch_page(self, region: str, year: int, cpv_prefix: str, page: int = 0) -> dict:
        """Ambil satu halaman hasil pencarian tender dari opentender.net."""
        params = {
            "country": "ID",
            "region": region,
            "year": year,
            "cpvs": f"{cpv_prefix}*",
            "page": page,
            "pageSize": self.config.get("page_size", 100),
        }
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"Opentender fetch gagal (page={page}): {e}")
            return {}

    def scrape(self, region: str, years: list[int], cpv_codes: list[str]) -> pd.DataFrame:
        """
        Scrape semua data tender untuk kombinasi region/year/cpv.
        Mengembalikan DataFrame mentah.
        """
        records = []
        for year in years:
            for cpv in cpv_codes:
                page = 0
                logger.info(f"Scraping opentender — region={region}, year={year}, CPV={cpv}")
                while True:
                    data = self.fetch_page(region, year, cpv, page)
                    hits = data.get("data", {}).get("hits", [])
                    if not hits:
                        break
                    for tender in hits:
                        records.extend(self._parse_tender(tender))
                    total = data.get("data", {}).get("total", 0)
                    fetched = (page + 1) * self.config.get("page_size", 100)
                    if fetched >= total:
                        break
                    page += 1
                    time.sleep(self.delay)
        df = pd.DataFrame(records)
        logger.info(f"Opentender: {len(df)} baris data dikumpulkan")
        return df

    def _parse_tender(self, tender: dict) -> list[dict]:
        """Parse satu objek tender menjadi list baris per peserta."""
        rows = []
        tender_id = tender.get("id", "")
        title = tender.get("title", "")
        date = tender.get("date", "")
        buyers = tender.get("buyers", [{}])
        instansi = buyers[0].get("name", "") if buyers else ""

        for lot in tender.get("lots", []):
            for bid in lot.get("bids", []):
                for bidder in bid.get("bidders", []):
                    rows.append({
                        "tender_id"       : tender_id,
                        "judul"           : title,
                        "instansi"        : instansi,
                        "tanggal"         : date,
                        "nama_asli"       : bidder.get("name", ""),
                        "nilai_penawaran" : bid.get("price", {}).get("netAmount", 0),
                        "is_winner"       : bool(bid.get("isWinning", False)),
                        "nilai_hps"       : 0.0,
                        "sumber"          : "opentender",
                    })
        return rows


# ─────────────────────────────────────────────
# Scraper — LKPP API (TRD 2.1.1 Target 2)
# ─────────────────────────────────────────────

class LKPPScraper:
    """Scraper untuk API LKPP (lpse.lkpp.go.id)."""

    BASE_URL = LKPP_BASE_URL

    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TenderGuard/1.0 (Research — GEMASTIK)",
        })
        self.delay = config.get("rate_limit_delay", 0.1)

    def fetch_page(self, lpse_id: str, tahun: int, start: int = 0, length: int = 100) -> dict:
        """Ambil satu halaman data lelang dari LPSE tertentu."""
        params = {
            "idLpse": lpse_id,
            "tahunAnggaran": tahun,
            "draw": 1,
            "start": start,
            "length": length,
        }
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"LKPP fetch gagal (lpse={lpse_id}, start={start}): {e}")
            return {}

    def scrape(self, lpse_ids: list[str], years: list[int]) -> pd.DataFrame:
        """Scrape semua data dari daftar LPSE ID dan tahun tertentu."""
        records = []
        page_size = 100
        for lpse_id in lpse_ids:
            for year in years:
                logger.info(f"Scraping LKPP — lpse_id={lpse_id}, tahun={year}")
                start = 0
                while True:
                    data = self.fetch_page(lpse_id, year, start, page_size)
                    rows_data = data.get("data", [])
                    if not rows_data:
                        break
                    for row in rows_data:
                        records.append(self._parse_row(row, lpse_id, year))
                    total = data.get("recordsTotal", 0)
                    start += page_size
                    if start >= total:
                        break
                    time.sleep(self.delay)
        df = pd.DataFrame(records)
        logger.info(f"LKPP: {len(df)} baris data dikumpulkan")
        return df

    def _parse_row(self, row: dict, lpse_id: str, year: int) -> dict:
        """Parse satu baris data LKPP."""
        return {
            "tender_id"       : f"LKPP-{lpse_id}-{row.get('kode_tender', '')}",
            "judul"           : row.get("nama_paket", ""),
            "instansi"        : row.get("nama_instansi", ""),
            "tanggal"         : row.get("tgl_pembuatan", ""),
            "nama_asli"       : row.get("nama_penyedia", ""),
            "nilai_penawaran" : row.get("harga_penawaran", 0),
            "is_winner"       : row.get("status_pemenang", False),
            "nilai_hps"       : row.get("hps", 0),
            "sumber"          : "lkpp",
        }


# ─────────────────────────────────────────────
# Scraper — inaproc.id (TRD 2.1.1 Target 3)
# ─────────────────────────────────────────────

class InaprocScraper:
    """Scraper untuk inaproc.id sebagai sumber pelengkap."""

    BASE_URL = INAPROC_BASE_URL

    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.delay = config.get("rate_limit_delay", 0.1)

    def fetch(self, satker: str, tahun: int) -> list[dict]:
        """Ambil data kontrak dari inaproc.id."""
        params = {
            "tahun": tahun,
            "satker": satker,
            "format": "json",
        }
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except requests.RequestException as e:
            logger.warning(f"InaProc fetch gagal (satker={satker}): {e}")
            return []

    def scrape(self, satker_ids: list[str], years: list[int]) -> pd.DataFrame:
        """Scrape data dari daftar satuan kerja dan tahun."""
        records = []
        for satker in satker_ids:
            for year in years:
                logger.info(f"Scraping InaProc — satker={satker}, tahun={year}")
                data = self.fetch(satker, year)
                for row in data:
                    records.append({
                        "tender_id"       : f"INAPROC-{row.get('kd_rup', '')}",
                        "judul"           : row.get("nama_paket", ""),
                        "instansi"        : row.get("nama_satker", ""),
                        "tanggal"         : row.get("tgl_kontrak", ""),
                        "nama_asli"       : row.get("nama_penyedia", ""),
                        "nilai_penawaran" : row.get("nilai_kontrak", 0),
                        "is_winner"       : True,  # InaProc hanya mencatat pemenang
                        "nilai_hps"       : row.get("nilai_pagu", 0),
                        "sumber"          : "inaproc",
                    })
                time.sleep(self.delay)
        df = pd.DataFrame(records)
        logger.info(f"InaProc: {len(df)} baris data dikumpulkan")
        return df


# ─────────────────────────────────────────────
# Pipeline Utama
# ─────────────────────────────────────────────

class DataIngestionPipeline:
    """
    Pipeline orkestrasi untuk semua sumber data.
    Menggabungkan, menormalisasi, dan menyimpan ke SQLite.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.data_cfg = self.cfg.get("data", {})

        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)

        self.db_conn = init_database("data/tender.db")
        self.opentender = OpentenderScraper(self.data_cfg)
        self.lkpp       = LKPPScraper(self.data_cfg)
        self.inaproc    = InaprocScraper(self.data_cfg)

    def run(
        self,
        use_opentender: bool = True,
        use_lkpp: bool = False,
        use_inaproc: bool = False,
        lpse_ids: list[str] | None = None,
        satker_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Jalankan pipeline ingestion secara menyeluruh.
        Mengembalikan DataFrame gabungan yang sudah dinormalisasi.
        """
        region  = self.data_cfg.get("target_region", "DKI Jakarta")
        years   = self.data_cfg.get("target_years", [2023])
        cpv     = self.data_cfg.get("cpv_codes", ["45"])
        min_val = self.data_cfg.get("min_tender_value", 0)

        frames = []

        if use_opentender:
            df_ot = self.opentender.scrape(region, years, cpv)
            df_ot.to_csv("data/raw/opentender_raw.csv", index=False)
            frames.append(df_ot)

        if use_lkpp and lpse_ids:
            df_lk = self.lkpp.scrape(lpse_ids, years)
            df_lk.to_csv("data/raw/lkpp_raw.csv", index=False)
            frames.append(df_lk)

        if use_inaproc and satker_ids:
            df_ip = self.inaproc.scrape(satker_ids, years)
            df_ip.to_csv("data/raw/inaproc_raw.csv", index=False)
            frames.append(df_ip)

        if not frames:
            logger.error("Tidak ada sumber data yang aktif. Gunakan flag use_opentender/use_lkpp/use_inaproc.")
            return pd.DataFrame()

        # Gabungkan semua sumber
        df = pd.concat(frames, ignore_index=True)
        logger.info(f"Total baris sebelum cleaning: {len(df)}")

        # Cleaning dasar
        df = df.dropna(subset=["tender_id", "nama_asli"])
        df = df[df["tender_id"].str.strip() != ""]
        df["nilai_penawaran"] = pd.to_numeric(df["nilai_penawaran"], errors="coerce").fillna(0)
        df["nilai_hps"] = pd.to_numeric(df["nilai_hps"], errors="coerce").fillna(0)

        # Filter nilai minimum
        if min_val > 0:
            df = df[df["nilai_hps"] >= min_val]

        # Normalisasi nama vendor
        logger.info("Melakukan deduplication nama vendor...")
        all_names = df["nama_asli"].unique().tolist()
        canonical_map = deduplicate_vendors(all_names, threshold=85)
        df["vendor_id"] = df["nama_asli"].map(canonical_map)

        # Ranking penawaran per tender
        df["rank"] = df.groupby("tender_id")["nilai_penawaran"].rank(ascending=True).astype(int)

        logger.info(f"Total baris setelah cleaning: {len(df)}")
        logger.info(f"Jumlah vendor unik: {df['vendor_id'].nunique()}")
        logger.info(f"Jumlah tender unik: {df['tender_id'].nunique()}")

        # Simpan ke CSV processed
        df.to_csv("data/processed/peserta_clean.csv", index=False)

        # Simpan ke SQLite
        self._save_to_db(df)

        return df

    def _save_to_db(self, df: pd.DataFrame):
        """Simpan DataFrame ke tabel SQLite."""
        cur = self.db_conn.cursor()

        # Upsert tender
        tenders = df[["tender_id", "judul", "instansi", "tanggal", "nilai_hps", "sumber"]].drop_duplicates("tender_id")
        for _, row in tqdm(tenders.iterrows(), total=len(tenders), desc="Menyimpan tender"):
            cur.execute("""
                INSERT OR REPLACE INTO tender (tender_id, judul, instansi, tanggal, nilai_hps, sumber)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (row.tender_id, row.judul, row.instansi, row.tanggal, row.nilai_hps, row.sumber))

        # Upsert vendor
        vendors = df[["vendor_id", "nama_asli"]].drop_duplicates("vendor_id")
        for _, row in tqdm(vendors.iterrows(), total=len(vendors), desc="Menyimpan vendor"):
            cur.execute("""
                INSERT OR IGNORE INTO vendor (vendor_id, nama_canonical)
                VALUES (?, ?)
            """, (row.vendor_id, row.vendor_id))

        # Insert peserta
        cur.execute("DELETE FROM peserta")  # Reset agar tidak duplikat saat re-run
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Menyimpan peserta"):
            cur.execute("""
                INSERT INTO peserta (tender_id, vendor_id, nama_asli, nilai_penawaran, is_winner, rank)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row.tender_id, row.vendor_id, row.nama_asli,
                row.nilai_penawaran, bool(row.is_winner), int(row["rank"])
            ))

        self.db_conn.commit()
        logger.info("Data berhasil disimpan ke SQLite.")

    def load_from_db(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load data dari SQLite untuk digunakan modul berikutnya."""
        df_tender  = pd.read_sql("SELECT * FROM tender",  self.db_conn)
        df_peserta = pd.read_sql("SELECT * FROM peserta", self.db_conn)
        df_peserta = df_peserta.merge(
            df_tender[["tender_id", "nilai_hps"]], on="tender_id", how="left"
        )
        return df_tender, df_peserta


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TenderGuard — Modul 1: Data Ingestion"
    )
    parser.add_argument("--config", default="config.yaml", help="Path ke file konfigurasi")
    parser.add_argument("--opentender", action="store_true", default=True, help="Aktifkan scraper opentender.net")
    parser.add_argument("--lkpp", action="store_true", default=False, help="Aktifkan scraper LKPP API")
    parser.add_argument("--inaproc", action="store_true", default=False, help="Aktifkan scraper inaproc.id")
    parser.add_argument("--lpse-ids", nargs="+", default=[], help="Daftar kode LPSE untuk scraper LKPP")
    parser.add_argument("--satker-ids", nargs="+", default=[], help="Daftar kode satker untuk InaProc")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = DataIngestionPipeline(config_path=args.config)
    df = pipeline.run(
        use_opentender=args.opentender,
        use_lkpp=args.lkpp,
        use_inaproc=args.inaproc,
        lpse_ids=args.lpse_ids,
        satker_ids=args.satker_ids,
    )
    print(f"\n✅ Ingestion selesai. Total {len(df)} baris data dikumpulkan.")
