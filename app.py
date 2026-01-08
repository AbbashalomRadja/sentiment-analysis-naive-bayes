from __future__ import annotations
import re, gc
from pathlib import Path
from datetime import datetime
from typing import Dict
from flask import (
    Flask, render_template, request, redirect, url_for, flash, send_file, session
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps 

import pandas as pd
from unidecode import unidecode
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import mysql.connector
from imblearn.over_sampling import RandomOverSampler

# Konfigurasi koneksi ke database MySQL


db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',        
    'database': 'sentimen_db'
}
def get_db_connection():
    return mysql.connector.connect(**db_config)

# === Bagian LOGIN, REGISTER, LOGOUT ===========
app = Flask(__name__)
app.secret_key = "super-secret-key"

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if not username or not password:
            flash("Username dan password harus diisi.", "warning")
            return redirect(url_for("register"))

        conn = get_db_connection()
        cursor = conn.cursor()

        # Cek apakah username sudah ada
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            flash("Username sudah digunakan, silakan pilih yang lain.", "danger")
            return redirect(url_for("register"))

        # Simpan user baru
        password_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (username, password_hash)
        )
        conn.commit()
        cursor.close()
        conn.close()

        flash("Registrasi berhasil! Silakan login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Selamat datang, {user['username']}!", "success")
            return redirect(url_for("index"))
        else:
            flash("Username atau password salah.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Anda telah logout.", "info")
    return redirect(url_for("login"))

@app.before_request
def require_login():
    # Endpoint yang boleh diakses tanpa login
    allowed_routes = ["login", "register", "static"]

    # Jika belum login dan bukan di halaman yang diizinkan → redirect ke login
    if request.endpoint not in allowed_routes and "user_id" not in session:
        return redirect(url_for("login"))


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Silakan login terlebih dahulu untuk mengakses halaman ini.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def insert_to_db(dataframe):
    """Simpan data dari DataFrame ke tabel dataset_train di MySQL."""
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    for _, row in dataframe.iterrows():
        cursor.execute(
            "INSERT INTO dataset_train (teks, label) VALUES (%s, %s)",
            (row['teks'], row['label'])
        )
    conn.commit()
    cursor.close()
    conn.close()

def load_from_db(limit=100):
    """Ambil data dari tabel dataset_train untuk ditampilkan di halaman upload."""
    try:
        conn = mysql.connector.connect(**db_config)
        query = "SELECT teks, label FROM dataset_train ORDER BY id DESC LIMIT %s"
        df = pd.read_sql(query, conn, params=(limit,))
        conn.close()
        return df
    except Exception as e:
        print("Gagal mengambil data dari database:", e)
        return pd.DataFrame(columns=["teks", "label"])



# ---------- Helpers: CSV robust loader & normalisasi kolom ----------
TEXT_COL_ALIASES = {
    # umum
    "teks", "text", "full_text", "clean_text", "text_clean",
    "tweet", "tweets", "post", "status", "message",
    # indo/variasi
    "konten", "isi", "isi_tweet", "kalimat", "ulasan", "review", "caption", "berita", "deskripsi"
}
LABEL_COL_ALIASES = {
    "label", "labels", "labeling",
    "sentimen", "sentiment", "polaritas", "polarity",
    "kategori", "category", "kelas", "class"
}

def load_csv_flex(path: Path) -> pd.DataFrame:
    """Coba baca CSV dengan beberapa encoding & delimiter umum (Excel/Sheets)."""
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] >= 1:
                    return df
            except Exception:
                continue
    return pd.read_csv(path)

def _find_col(df: pd.DataFrame, aliases: set[str]) -> str | None:
    """Cari kolom berdasarkan alias (case-insensitive, ignore spasi/underscore)."""
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    norm = {re.sub(r"[\s_]+", "", str(c).lower()): c for c in df.columns}
    for a in aliases:
        key = re.sub(r"[\s_]+", "", a.lower())
        if key in norm:
            return norm[key]
    return None

def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, str | None]:
    """
    Rename kolom alias ke standar: 'teks' dan 'label'.
    Return (df, original_text_col, original_label_col) untuk ditampilkan.
    """
    df.columns = [str(c).strip() for c in df.columns]

    text_col_orig  = _find_col(df, TEXT_COL_ALIASES)
    label_col_orig = _find_col(df, LABEL_COL_ALIASES)

    if text_col_orig and text_col_orig != "teks":
        df.rename(columns={text_col_orig: "teks"}, inplace=True)
    if label_col_orig and label_col_orig != "label":
        df.rename(columns={label_col_orig: "label"}, inplace=True)

    return df, text_col_orig, label_col_orig

import csv, io

def robust_read_csv(path):
    """Baca CSV dengan deteksi delimiter & gabungkan kolom tidak dikenal."""
    raw = Path(path).read_text(encoding="utf-8", errors="ignore").replace("\ufeff", "")
    # Deteksi pemisah (pakai ';' kalau koma terlalu banyak)
    sep = "," if raw.count(",") > raw.count(";") else ";"

    df = pd.read_csv(io.StringIO(raw), sep=sep, quoting=csv.QUOTE_ALL, engine="python", on_bad_lines="skip")

    # Gabungkan kolom tak bernama (kalau ada) jadi satu teks
    unnamed = [c for c in df.columns if "unnamed" in c.lower()]
    if unnamed:
        df[df.columns[0]] = df[df.columns[0]].astype(str) + " " + df[unnamed].astype(str).agg(" ".join, axis=1)
        df.drop(columns=unnamed, inplace=True, errors="ignore")

    # Normalisasi nama kolom
    df.columns = [c.strip().lower() for c in df.columns]
    if "text" in df.columns:
        df.rename(columns={"text": "teks"}, inplace=True)

    return df


def smart_read_csv(path):
    """
    Membaca CSV yang punya tanda kutip, koma di dalam teks, atau format tidak rapi.
    Otomatis deteksi delimiter dan perbaiki encoding.
    """
    import io, csv

    # Baca mentah untuk hapus karakter aneh
    raw = Path(path).read_text(encoding="utf-8", errors="ignore").replace("\ufeff", "")

    # Deteksi delimiter otomatis
    sep = "," if raw.count(",") > raw.count(";") else ";"

    # Gunakan quoting agar koma di dalam teks tidak memecah kolom
    try:
        df = pd.read_csv(
            io.StringIO(raw),
            sep=sep,
            engine="python",
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="skip"
        )
    except Exception:
        df = pd.read_csv(io.StringIO(raw), sep=sep, engine="python", on_bad_lines="skip")

    # Pastikan nama kolom lowercase dan ganti 'text' jadi 'teks'
    df.columns = [c.strip().lower() for c in df.columns]
    if "text" in df.columns:
        df.rename(columns={"text": "teks"}, inplace=True)

    return df

# -----------------------
# Paths & App init
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR  = BASE_DIR / "uploads"
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
for d in (UPLOAD_DIR, MODELS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
STATIC_DIR = BASE_DIR / "static"
PLOTS_DIR = STATIC_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".csv"}

# -----------------------
# Dashboard state
# -----------------------
APP_STATE: Dict[str, object] = {
    "n_train": 0,
    "n_test": 0,
    "train_counts": {"positif": 0, "negatif": 0, "netral": 0},
    "test_counts":  {"positif": 0, "negatif": 0, "netral": 0},
    "last_train_acc": None,
    "last_train_report": None,
}

# ===== tambahan: simpan/muat metrik training & state =====
import json
TRAIN_METRICS_PATH = MODELS_DIR / "train_metrics.json"

def save_train_metrics(metrics: dict):
    try:
        with open(TRAIN_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[WARN] gagal menyimpan train metrics:", e)

# ------------------------------------
# FUNGSI INTERPRETASI OTOMATIS MODEL
# ------------------------------------
def generate_interpretasi(accuracy, class_counts):
    # Cari kelas dominan
    dominant_class = max(class_counts, key=class_counts.get)
    dominant_count = class_counts[dominant_class]

    # Ambil total data
    total = sum(class_counts.values())

    # Buat interpretasi pendek
    interpretasi = (
        f"Akurasi model saat ini adalah {accuracy:.2f}. "
        f"Dataset didominasi oleh kelas '{dominant_class}' dengan {dominant_count} data "
        f"dari total {total} data. "
        "Dominasi ini membuat model lebih mudah mengenali kelas tersebut. "
        "Menambah data pada kelas lain dapat membantu meningkatkan performa model."
    )

    return interpretasi

def load_train_metrics() -> dict | None:
    try:
        if TRAIN_METRICS_PATH.exists():
            with open(TRAIN_METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print("[WARN] gagal memuat train metrics:", e)
    return None

# Muat metrics saat app start (jika ada) agar APP_STATE tampilkan data lama
_loaded_metrics = load_train_metrics()
if _loaded_metrics:
    APP_STATE["last_train_acc"] = _loaded_metrics.get("accuracy_float") or APP_STATE.get("last_train_acc")
    APP_STATE["last_train_report"] = _loaded_metrics.get("report_text") or APP_STATE.get("last_train_report")
    APP_STATE["last_train_metrics_file"] = str(TRAIN_METRICS_PATH)


# Text preprocessing
import re
from unidecode import unidecode
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Regex dasar
URL_RE        = re.compile(r"http\S+|www\.\S+")
MENTION_RE    = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE    = re.compile(r"#\w+")
EMOJI_RE      = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]", re.UNICODE)
PUNCT_RE      = re.compile(r"[^\w\s]")
REPEAT_CHAR   = re.compile(r"(.)\1{2,}")       # coooool -> coo
MULTISPACE_RE = re.compile(r"\s+")
DIGIT_RUN_RE  = re.compile(r"\b([A-Za-z]+)\d+\b")  # aksi2 -> aksi

# Stemmer & stopword
stemmer = StemmerFactory().create_stemmer()
_base_stop = set(StopWordRemoverFactory().get_stop_words())

# Pertahankan negasi & intensitas agar positif≠netral
_KEEP = {"tidak", "bukan", "belum", "padahal", "tetapi", "sangat", "lebih", "sekali", "cukup", "lumayan", "banget"}
STOPWORDS = (_base_stop - _KEEP)

# Slang / normalisasi umum (diperluas)
SLANG_MAP = {
    r"\bga?k+\b": "tidak", r"\bngg?ak\b": "tidak", r"\bgk\b": "tidak", r"\bt?dk\b": "tidak",
    r"\bgada\b": "tidak ada", r"\bgpp\b": "tidak apa", r"\bblm\b": "belum",
    r"\bs?dh\b": "sudah", r"\budh\b": "sudah", r"\buda(h)?\b": "sudah",
    r"\bjd[i]?\b": "jadi", r"\bjg\b": "juga", r"\bttp\b": "tetap",
    r"\btp|tpi\b": "tapi", r"\bkrn(a)?\b": "karena",
    r"\bklo|klu|kl\b": "kalau", r"\butk\b": "untuk", r"\bbuat\b": "untuk",
    r"\bdg(n)?\b": "dengan", r"\bdr\b": "dari", r"\bdlm\b": "dalam",
    r"\bsm|sma\b": "sama", r"\bbkn\b": "bukan", r"\bbgt\b": "banget", r"\bsgt\b": "sangat",
    r"\bmsh\b": "masih", r"\bdoang\b": "saja", r"\baj?a?\b": "saja",
    r"\bcmn\b": "cuma", r"\bmn\b": "mana", r"\bdmn\b": "dimana", r"\bpd\b": "pada", r"\bsbg\b": "sebagai",
    r"\btrs\b": "terus", r"\bspy\b": "supaya", r"\bspt\b": "seperti",
    r"\byg\b": "yang", r"\bdpt\b": "dapat", r"\bkrg\b": "kurang", r"\bbbrp\b": "beberapa",
    r"\bpls|plis|please\b": "tolong", r"\bmakasih|mksh|thx|thanks\b": "terima kasih",
    r"\bgua\b|\bgw\b": "saya", r"\bsy\b": "saya", r"\bkm\b": "kamu", r"\blu|loe\b": "kamu",
    r"\bkte\b": "kita", r"\bmrk\b": "mereka",
    r"\bora\b": "tidak", r"\bwes|wis\b": "sudah", r"\bopo\b": "apa", r"\bjo\b": "jangan",
    r"\bwkwk+\b|\bxixi+\b|\bhehe+\b|\bhaha+\b": "haha",
    r"\bmantap+\b|\bmantul+\b|\bkeren+\b|\bthe\s+best\b": "bagus", r"\btop\b": "bagus", r"\bgood\b": "bagus",
    r"\bngurusin\b": "mengurus", r"\bngabisin\b": "menghabiskan", r"\bampas\b": "buruk",
}

# Domain map (negatif/netral + tambah positif supaya tidak “larut”)
DOMAIN_MAP = {
    # negatif/kriminal
    r"\b(merampok|perampokan|rampok)\b": "perampokan",
    r"\b(dirampas|perampasan|rampas)\b": "perampasan",
    r"\b(dijarah|penjarahan|menjarah)\b": "penjarahan",
    r"\b(kerusuhan|bentro(k|kan)|huru\s*hara)\b": "kerusuhan",
    r"\b(pembakaran|dibakar|membakar)\b": "pembakaran",
    r"\b(perusakan|dirusak|merusak)\b": "perusakan",
    r"\b(pidana|kriminal|tindak\s*pidana)\b": "kriminalitas",
    r"\b(ilegal|tanpa\s*izin|tidak\s*sah)\b": "ilegal",
    # netral hukum/aset
    r"\b(pengambilalihan|mengambil\s*alih)\b": "pengambilalihan aset",
    r"\b(pengembalian\s*aset)\b": "pengembalian aset",
    r"\b(pengamanan\s*aset|penertiban\s*aset)\b": "pengamanan aset",
    r"\b(pengelolaan\s*aset)\b": "pengelolaan aset",
    # positif
    r"\b(apresiasi|mengapresiasi)\b": "apresiasi",
    r"\b(puas|memuaskan)\b": "puas",
    r"\b(suka|menyukai)\b": "suka",
    r"\b(bagus|terbaik|baik|mantap)\b": "bagus",
    r"\b(senang|bahagia)\b": "senang",
}

# Kata yang tidak boleh di-stem (agar makna tidak rusak)
PRESERVE = {
    "penjarahan", "perampokan", "perampasan", "kerusuhan", "pembakaran", "perusakan", "kriminalitas",
    "pengelolaan", "pengamanan", "pengembalian", "pengambilalihan", "aset",
    "apresiasi", "puas", "suka", "bagus", "senang",
    # kata umum sering rusak
    "tangan", "orang", "rumah", "negara", "hukum", "uang", "menteri", "masyarakat"
}

# Compile mapping
COMPILED_SLANG  = [(re.compile(p), r) for p, r in SLANG_MAP.items()]
COMPILED_DOMAIN = [(re.compile(p), r) for p, r in DOMAIN_MAP.items()]

def _apply_map(text: str, compiled_pairs):
    for pat, rep in compiled_pairs:
        text = pat.sub(rep, text)
    return text

def _light_norm(text: str) -> str:
    t = unidecode(text or "")
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    t = t.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = re.sub(r'"+', " ", t)      # hilangkan kutip ganda ""...
    t = PUNCT_RE.sub(" ", t)
    t = REPEAT_CHAR.sub(r"\1\1", t)
    t = DIGIT_RUN_RE.sub(r"\1", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

def indo_clean_final(text: str) -> str:
    """Cleaner FINAL: jaga makna, kuat untuk kalimat pendek, dan tidak over-clean."""
    if not isinstance(text, str) or not text.strip():
        return ""
    t = _light_norm(text)
    t = _apply_map(t, COMPILED_SLANG)    # slang dulu
    t = _apply_map(t, COMPILED_DOMAIN)   # domain kemudian
    # stopword removal (negasi & intensitas dijaga)
    tokens = [w for w in t.split() if w not in STOPWORDS]
    # stemming dengan preserve
    out = []
    for w in tokens:
        if w in PRESERVE:
            out.append(w)
        else:
            out.append(stemmer.stem(w))
    t = " ".join(out)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

# ===== Positive hint builder & loader =====
from collections import Counter
import json, os, re

POS_HINT_PATH = MODELS_DIR / "positive_hint.json"

# Seed kamus positif (filter awal)
SEED_POS = set("""
bagus baik keren mantap mantul hebat top terbaik salut dukung support setuju puas suka senang
apresiasi damai kondusif aman lancar luar biasa oke ok recommended rekomendasi terimakasih terima kasih
""".split())

_token_re = re.compile(r"^[a-z]+$")

def build_positive_hint_from_texts(texts_pos: list[str], texts_nonpos: list[str], min_freq=3, max_n=40):
    """Ambil kata yang cukup sering & khas di kelas positif."""
    pos_c = Counter()
    non_c = Counter()

    for s in texts_pos:
        for w in str(s).split():
            if len(w) < 3 or not _token_re.match(w):
                continue
            pos_c[w] += 1

    for s in texts_nonpos:
        for w in str(s).split():
            if len(w) < 3 or not _token_re.match(w):
                continue
            non_c[w] += 1

    candidates = []
    for w, cp in pos_c.items():
        if cp < min_freq:
            continue
        cn = non_c.get(w, 0)
        ratio = (cp + 1) / (cn + 1)
        if (w in SEED_POS) and (ratio >= 1.1):
            candidates.append((w, cp, cn, ratio))

    candidates.sort(key=lambda x: (x[3], x[1]), reverse=True)
    return [w for w, _, _, _ in candidates[:max_n]]

def save_positive_hint(words: list[str], path=POS_HINT_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(words, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[WARN] gagal menyimpan positive hint:", e)

def load_positive_hint(path=POS_HINT_PATH) -> set[str]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return set(json.load(f))
    except Exception as e:
        print("[WARN] gagal load positive hint:", e)

    # fallback aman
    return {"keren", "salut", "damai", "bagus"}

# dipake di cek kalimat
GLOBAL_POSITIVE_HINT = load_positive_hint()

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT

# -----------------------
# Routes
# -----------------------
@app.route("/")
@app.route("/index")
@app.route("/index.html")
@login_required
def index():
    # kalau belum login, langsung arahkan ke halaman login
    if "user_id" not in session:
        return redirect(url_for("login"))
    # kalau sudah login, tampilkan dashboard
    return render_template("index.html", state=APP_STATE)
# ---- Upload training CSV (kolom: teks, label)

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    preview_html, info = None, {}

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Upload baru
    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            flash("Pilih file CSV terlebih dahulu.", "warning")
            return redirect(url_for("upload"))
        if not allowed_file(f.filename):
            flash("Hanya mendukung file .csv", "danger")
            return redirect(url_for("upload"))

        filename = secure_filename(f.filename)
        save_path = UPLOAD_DIR / filename
        f.save(save_path)

        # Baca file CSV dengan format aman (tidak autodetect)
        try:
            df = pd.read_csv(
                save_path,
                encoding="utf-8",
                sep=",",
                quotechar='"',
                engine="python",
                on_bad_lines="skip"
            )
            print(f"[DEBUG] Kolom terbaca: {df.columns.tolist()} — Total baris: {len(df)}")
        except Exception as e:
            flash(f"Gagal membaca CSV: {e}", "danger")
            return redirect(url_for("upload"))

        # Normalisasi kolom
        df, text_col_orig, label_col_orig = normalize_columns(df)
        if not text_col_orig or not label_col_orig:
            flash("Kolom teks/label tidak ditemukan.", "danger")
            return redirect(url_for("upload"))

        df = df.dropna(subset=["teks", "label"])
        total_rows = len(df)

        # Simpan info ke database
        cursor.execute("""
            INSERT INTO datasets (filename, total_rows, upload_time, status)
            VALUES (%s, %s, NOW(), 'raw')
        """, (filename, total_rows))
        conn.commit()

        flash(f"Upload OK. File '{filename}' berhasil disimpan ke database ({total_rows} baris).", "success")

    # Ambil semua dataset dari database
    cursor.execute("SELECT * FROM datasets ORDER BY id DESC")
    datasets = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template("upload.html", datasets=datasets, preview_html=preview_html, info=info)


# DETAIL DATASET
@app.route("/dataset/<int:id>")
@login_required
def dataset_detail(id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Ambil data dataset dari tabel
    cursor.execute("SELECT * FROM datasets WHERE id = %s", (id,))
    dataset = cursor.fetchone()

    if not dataset:
        flash("Dataset tidak ditemukan.", "warning")
        return redirect(url_for("upload"))

    # Baca file CSV dari folder uploads
    csv_path = UPLOAD_DIR / dataset["filename"]
    if not csv_path.exists():
        flash("File dataset tidak ditemukan di folder uploads.", "danger")
        return redirect(url_for("upload"))

    df = pd.read_csv(csv_path)
    preview_html = df.head(50).to_html(classes="table table-striped table-bordered", index=False)

    cursor.close()
    conn.close()

    return render_template("dataset_detail.html", dataset=dataset, preview_html=preview_html)


# CLEAN DATASET (Preprocessing)
from concurrent.futures import ThreadPoolExecutor
import time

@app.route("/dataset/<int:id>/clean")
@login_required
def dataset_clean(id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Ambil metadata dataset
    cursor.execute("SELECT * FROM datasets WHERE id = %s", (id,))
    dataset = cursor.fetchone()

    if not dataset:
        flash("Dataset tidak ditemukan.", "warning")
        return redirect(url_for("upload"))

    csv_path = UPLOAD_DIR / dataset["filename"]

    # 🧠 Jika dataset sudah bersih sebelumnya
    if dataset.get("status") == "cleaned" and dataset.get("cleaned_path"):
        flash(f"Dataset '{dataset['filename']}' sudah dibersihkan sebelumnya.", "info")
        APP_STATE["last_preprocessed_path"] = dataset["cleaned_path"]

        # Tetap update dashboard state agar tidak kosong
        try:
            df = pd.read_csv(dataset["cleaned_path"])
            if "label" in df.columns:
                counts = df["label"].astype(str).str.lower().value_counts().to_dict()
                APP_STATE["train_counts"] = {
                    "positif": int(counts.get("positif", 0)),
                    "negatif": int(counts.get("negatif", 0)),
                    "netral": int(counts.get("netral", 0)),
                }
                APP_STATE["n_train"] = len(df)
        except Exception as e:
            print(f"[WARNING] Gagal baca cleaned file: {e}")

        return redirect(url_for("preprocess"))

    # Jika belum clean, pastikan file ada
    if not csv_path.exists():
        flash("File dataset tidak ditemukan di folder uploads.", "danger")
        return redirect(url_for("upload"))

    # Baca file CSV mentah (auto deteksi delimiter)
    try:
        df = robust_read_csv(csv_path)
    except Exception as e:
        flash(f"Gagal membaca CSV: {e}", "danger")
        return redirect(url_for("upload"))

    # Normalisasi kolom agar kolom teks selalu bernama 'teks'
    df, text_col, label_col = normalize_columns(df)
    if not text_col:
        flash("Kolom teks tidak ditemukan di file dataset.", "danger")
        return redirect(url_for("upload"))

    # Mulai proses cleaning
    start = time.time()
    print(f"[INFO] Cleaning dimulai untuk {len(df)} baris...")

    # Gunakan multithreading agar cepat
    with ThreadPoolExecutor(max_workers=4) as executor:
        df["clean"] = list(executor.map(indo_clean_final,df["teks"].astype(str)))
    durasi = time.time() - start
    print(f"[INFO] Cleaning selesai dalam {durasi:.2f} detik.")

    # Simpan hasil cleaning ke folder outputs
    output_path = OUTPUTS_DIR / f"cleaned_{dataset['filename']}"
    df.to_csv(output_path, index=False)

    # Update status dan path hasil ke database
    cursor.execute("""
        UPDATE datasets 
        SET status = %s, cleaned_path = %s 
        WHERE id = %s
    """, ("cleaned", str(output_path), id))
    conn.commit()

    cursor.close()
    conn.close()

    # Simpan ke APP_STATE untuk halaman preprocess
    APP_STATE["last_preprocessed_path"] = str(output_path)

    # ---------------- Update Dashboard State ----------------
    if "label" in df.columns:
        counts = df["label"].astype(str).str.lower().value_counts().to_dict()
        APP_STATE["train_counts"] = {
            "positif": int(counts.get("positif", 0)),
            "negatif": int(counts.get("negatif", 0)),
            "netral":  int(counts.get("netral", 0)),
        }
        APP_STATE["n_train"] = len(df)
    else:
        APP_STATE["train_counts"] = {"positif": 0, "negatif": 0, "netral": 0}
        APP_STATE["n_train"] = len(df)

    flash(f"✅ Dataset '{dataset['filename']}' berhasil dibersihkan dalam {durasi:.1f} detik dan disimpan.", "success")
    return redirect(url_for("preprocess"))

@app.route("/dataset/<int:id>/delete", methods=["POST"])
@login_required
def dataset_delete(id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM datasets WHERE id = %s", (id,))
    dataset = cursor.fetchone()

    if not dataset:
        flash("Dataset tidak ditemukan.", "warning")
        return redirect(url_for("upload"))

    # Hapus file CSV dan hasil cleaned jika ada
    csv_path = UPLOAD_DIR / dataset["filename"]
    if csv_path.exists():
        csv_path.unlink()  # hapus file asli

    cleaned_path = dataset.get("cleaned_path")
    if cleaned_path and Path(cleaned_path).exists():
        Path(cleaned_path).unlink()  # hapus file hasil clean

    # Hapus data dari database
    cursor.execute("DELETE FROM datasets WHERE id = %s", (id,))
    conn.commit()

    cursor.close()
    conn.close()

    flash(f"🗑️ Dataset '{dataset['filename']}' berhasil dihapus.", "success")
    return redirect(url_for("upload"))

# ---- Preprocess training -> simpan .xlsx dan update dashboard
@app.route("/preprocess", methods=["GET", "POST"])
@login_required
def preprocess():
    last_path = APP_STATE.get("last_preprocessed_path")
    df = None
    preview = None
    label_counts = {}

    # Jika sudah ada hasil cleaning
    if last_path and Path(last_path).exists():
        try:
            df = pd.read_csv(last_path)
            preview = df.head(10).to_html(classes="table table-sm table-striped", index=False)

            # Hitung distribusi label (kalau ada kolom label)
            if "label" in df.columns:
                label_counts = df["label"].str.lower().value_counts().to_dict()

        except Exception as e:
            flash(f"Gagal memuat file hasil cleaning: {e}", "warning")
            
    #  DETAIL MODE (lihat semua hasil)
    if request.args.get("view") == "detail" and df is not None:
        q = request.args.get("q", "")
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))

        # Filtering
        if q:
            mask = df.apply(lambda row: row.astype(str).str.contains(q, case=False, na=False).any(), axis=1)
            df_filtered = df[mask]
        else:
            df_filtered = df

        total = len(df_filtered)
        total_pages = max((total + per_page - 1) // per_page, 1)
        start = (page - 1) * per_page
        end = start + per_page
        df_page = df_filtered.iloc[start:end]

        table_html = df_page.to_html(classes="table table-hover table-bordered table-sm", index=False)

        return render_template(
            "preprocess.html",
            state=APP_STATE,
            preview_html=preview,
            label_counts=label_counts,
            table_html=table_html,
            total=total,
            total_pages=total_pages,
            page=page,
            per_page=per_page
        )        

    # Kalau user jalankan preprocessing ulang
    if request.method == "POST":
        if df is None:
            flash("Tidak ada dataset yang tersedia untuk preprocessing.", "warning")
            return redirect(url_for("upload"))

        df["clean2"] = df["clean"].astype(str).apply(indo_clean_final)
        out = df[["teks", "clean", "clean2", "label"]] if "label" in df.columns else df[["teks", "clean", "clean2"]]

        out_path = OUTPUTS_DIR / f"preprocessed_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        out.to_excel(out_path, index=False)

        APP_STATE["last_preprocessed_path"] = str(out_path)

        flash("Preprocessing ulang selesai. Anda dapat mengunduh hasilnya.", "success")
        preview = out.head(10).to_html(classes="table table-sm table-striped", index=False)

        if "label" in out.columns:
            label_counts = out["label"].str.lower().value_counts().to_dict()

    return render_template("preprocess.html", preview_html=preview, state=APP_STATE, label_counts=label_counts)

@app.route("/preprocess/detail")
@login_required
def preprocess_detail():
    path = APP_STATE.get("last_preprocessed_path")
    if not path or not Path(path).exists():
        flash("Belum ada hasil preprocessing untuk ditampilkan.", "warning")
        return redirect(url_for("preprocess"))

    q = request.args.get("q", "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    per_page = int(request.args.get("per_page", 50))
    per_page = min(max(per_page, 10), 500)

    # baca tabel
    try:
        if str(path).lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, engine="openpyxl")
    except Exception:
        df = pd.read_csv(path)

    pref_cols = ["teks", "clean", "clean2", "label"]
    cols_exist = [c for c in pref_cols if c in df.columns] or list(df.columns)
    df = df[cols_exist]

    if q:
        mask = False
        for c in cols_exist:
            mask = mask | df[c].astype(str).str.contains(q, case=False, na=False)
        df = df[mask]

    total = len(df)
    total_pages = max((total + per_page - 1) // per_page, 1)
    start = (page - 1) * per_page
    end = start + per_page
    df_page = df.iloc[start:end]

    table_html = df_page.to_html(
        classes="table table-hover table-bordered table-sm",
        index=False,
        escape=False
    )

    return render_template(
        "preprocess_detail.html",
        table_html=table_html,
        q=q,
        page=page,
        per_page=per_page,
        total=total,
        total_pages=total_pages,
    )


@app.route("/download/preprocessed")
def download_preprocessed():
    path = APP_STATE.get("last_preprocessed_path")
    if not path or not Path(path).exists():
        flash("Belum ada hasil preprocessing untuk diunduh.", "warning")
        return redirect(url_for("preprocess"))
    return send_file(path, as_attachment=True)

@app.route("/train", methods=["GET", "POST"])
@login_required
def train():
    path = APP_STATE.get("last_preprocessed_path")
    model_path = MODELS_DIR / "nb_tfidf_pipeline.joblib"

    # ================== GET REQUEST (tampilkan hasil lama jika tidak ada preprocessed) ==================
    if request.method == "GET":
        if (not path or not Path(path).exists()) and model_path.exists():
            metrics = load_train_metrics()

            # susun metrics lama
            if not metrics:
                metrics_display = {
                    "accuracy": f"{APP_STATE.get('last_train_acc') or 'N/A'}",
                    "report_text": APP_STATE.get("last_train_report"),
                    "cm_img": None,
                    "metrics_data": {}
                }
            else:
                metrics_display = {
                    "accuracy": metrics.get("accuracy_str", f"{APP_STATE.get('last_train_acc') or 'N/A'}"),
                    "report_text": metrics.get("report_text"),
                    "cm_img": metrics.get("cm_img"),
                    "metrics_data": metrics.get("metrics_data", {})
                }

            from datetime import datetime
            return render_template(
                "train.html",
                n_samples=APP_STATE.get("n_train", 0),
                state=APP_STATE,
                metrics=metrics_display,
                current_date=datetime.now().strftime("%d %b %Y"),
                info_msg="Menampilkan hasil training terakhir karena tidak ditemukan preprocessed file."
            )

    # ================== CEK PREPROCESSED ==================
    if not path or not Path(path).exists():
        flash("Jalankan preprocessing terlebih dahulu.", "warning")
        return redirect(url_for("preprocess"))

    # baca data file
    try:
        if str(path).lower().endswith(".xlsx"):
            df = pd.read_excel(path, engine="openpyxl")
        else:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)

    # normalisasi label
    y_all = df["label"].astype(str).str.lower().replace({
        "positive": "positif",
        "negative": "negatif",
        "neutral": "netral"
    })
    X_all = df["clean"].astype(str)

    metrics = None

    # ================== POST = PROSES TRAINING ==================
    if request.method == "POST":
        split_ratio = float(request.form.get("split_ratio", "0.8"))
        test_size = 1.0 - split_ratio

        X_tr, X_va, y_tr, y_va = train_test_split(
            X_all, y_all, test_size=test_size, random_state=42, stratify=y_all
        )

        # ===== Oversampling =====
        try:
            from imblearn.over_sampling import RandomOverSampler
        except Exception:
            flash("Library imbalanced-learn belum terpasang.", "danger")
            return redirect(url_for("train"))

        ros = RandomOverSampler(random_state=42)
        X_tr_df = pd.DataFrame({"text": X_tr.values})
        X_res, y_res = ros.fit_resample(X_tr_df, y_tr)

        if isinstance(X_res, pd.DataFrame):
            X_res_series = X_res.iloc[:, 0].astype(str)
        else:
            import numpy as _np
            X_res_series = pd.Series(_np.squeeze(X_res).astype(str))

        # ===== Pipeline + RandomizedSearchCV =====
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        from sklearn.feature_selection import SelectKBest, chi2
        from sklearn.naive_bayes import MultinomialNB, ComplementNB
        import time

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("select", SelectKBest(chi2)),
            ("clf", MultinomialNB())
        ])

        param_dist = {
            "tfidf__ngram_range": [(1,1), (1,2)],
            "tfidf__max_features": [10000, 20000],
            "tfidf__min_df": [1, 2],
            "tfidf__max_df": [0.8, 0.9, 1.0],
            "tfidf__sublinear_tf": [True],
            "select__k": [5000, 10000, "all"],
            "clf": [MultinomialNB(), ComplementNB()],
            "clf__alpha": [0.01, 0.1, 0.3, 0.5]
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        rs = RandomizedSearchCV(
            pipe, param_dist,
            n_iter=20, cv=cv, scoring="accuracy",
            n_jobs=-1, verbose=2, random_state=42
        )

        t0 = time.time()
        rs.fit(X_res_series, y_res)
        print(f"[INFO] RandomizedSearch selesai dalam {time.time()-t0:.1f} detik")

        best_pipe = rs.best_estimator_
        preds = best_pipe.predict(X_va)

        # ===== METRIK =====
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.utils.multiclass import unique_labels
        import numpy as np

        acc = accuracy_score(y_va, preds)
        report = classification_report(y_va, preds, output_dict=True, zero_division=0)

        labels = list(unique_labels(y_va))
        cm = confusion_matrix(y_va, preds, labels=labels)

        metrics_data = {lbl: report.get(lbl, {}) for lbl in labels}

        metrics = {
            "accuracy": f"{acc:.4f}",
            "report_text": classification_report(y_va, preds, digits=3, zero_division=0),
            "metrics_data": metrics_data,
            "cm": cm.tolist(),
            "labels": labels
        }

        # ===== CONFUSION MATRIX IMAGE =====
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from datetime import datetime
            from flask import url_for

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, cmap="Blues")
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(len(labels)),
                yticks=np.arange(len(labels)),
                xticklabels=labels,
                yticklabels=labels,
                xlabel="Prediksi",
                ylabel="Aktual",
                title="Confusion Matrix"
            )
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

            fig.tight_layout()
            cm_filename = f"cm_{ts}.png"
            fig.savefig(PLOTS_DIR / cm_filename, dpi=160)
            plt.close(fig)

            metrics["cm_img"] = url_for("static", filename=f"plots/{cm_filename}")

        except Exception as e:
            print("[WARN] Gagal render CM:", e)
            metrics["cm_img"] = None

        # ===== SIMPAN MODEL =====
        joblib.dump(best_pipe, MODELS_DIR / "nb_tfidf_pipeline.joblib")

        # ===== APP_STATE (hasil training) =====
        APP_STATE["last_train_acc"] = round(float(acc), 4)
        APP_STATE["last_train_report"] = metrics["report_text"]

        counts = y_all.value_counts().to_dict()
        APP_STATE["train_counts"] = {
            "positif": int(counts.get("positif", 0)),
            "negatif": int(counts.get("negatif", 0)),
            "netral": int(counts.get("netral", 0)),
        }
        APP_STATE["n_train"] = len(df)

        flash(f"Training selesai. Akurasi validasi: {acc:.4f}", "success")

        # ===== Positive Hint =====
        y_res_id = pd.Series(y_res).astype(str).str.lower()
        pos_texts = list(X_res_series[y_res_id == "positif"])
        nonpos_texts = list(X_res_series[y_res_id != "positif"])

        pos_hint = build_positive_hint_from_texts(pos_texts, nonpos_texts, min_freq=3, max_n=40)
        if len(pos_hint) < 3:
            pos_hint = list((set(pos_hint) | {"keren", "salut", "damai", "bagus"}))

        save_positive_hint(pos_hint)
        global GLOBAL_POSITIVE_HINT
        GLOBAL_POSITIVE_HINT = set(pos_hint)

        # ===== SIMPAN METRICS =====
        try:
            store = {
                "accuracy_str": metrics.get("accuracy"),
                "accuracy_float": float(acc),
                "report_text": metrics.get("report_text"),
                "metrics_data": metrics.get("metrics_data", {}),
                "cm": metrics.get("cm"),
                "labels": metrics.get("labels"),
                "cm_img": metrics.get("cm_img"),
                "timestamp": datetime.now().isoformat()
            }
            save_train_metrics(store)
        except Exception as e:
            print("[WARN] Gagal menyimpan metrics:", e)

        # ======================================================================
        # ================ TREND SENTIMEN PER MINGGU (MANUAL) ==================
        # ======================================================================
        try:
            df_tr = df.copy().reset_index(drop=True)

            # Range manual sesuai dataset kamu
            def assign_week(idx):
                if idx < 1012:
                    return "Week 1"
                elif idx < 1012 + 515:
                    return "Week 2"
                elif idx < 1012 + 515 + 517:
                    return "Week 3"
                else:
                    return "Week 4"

            df_tr["week"] = df_tr.index.map(assign_week)

            df_tr["label_norm"] = df_tr["label"].astype(str).str.lower().replace({
                "positive": "positif",
                "negative": "negatif",
                "neutral": "netral"
            })

            weekly_trend = (
                df_tr.groupby(["week", "label_norm"])
                .size()
                .reset_index(name="count")
                .pivot(index="week", columns="label_norm", values="count")
                .fillna(0)
                .to_dict(orient="index")
            )

            APP_STATE["weekly_trend"] = weekly_trend

        except Exception as e:
            print("[WARN] Gagal hitung trend manual:", e)
            APP_STATE["weekly_trend"] = None
        # ======================================================================
        interpretasi = generate_interpretasi(acc, APP_STATE["train_counts"])

        from datetime import datetime
        return render_template(
            "train.html",
            n_samples=len(X_all),
            state=APP_STATE,
            metrics=metrics,
            current_date=datetime.now().strftime("%d %b %Y"),
            info_msg=None,
            interpretasi=interpretasi
        )

    # ================== GET DEFAULT ==================
    from datetime import datetime
    return render_template( 
        "train.html",
        n_samples=len(X_all),
        state=APP_STATE,
        metrics=None,
        current_date=datetime.now().strftime("%d %b %Y"),
        info_msg=None
    )

# ---- Testing (upload CSV kolom: teks)
@app.route("/test", methods=["GET", "POST"])
@login_required
def test_page():
    model_path = MODELS_DIR / "nb_tfidf_pipeline.joblib"
    if not model_path.exists():
        flash("Silakan lakukan training terlebih dahulu.", "warning")
        return redirect(url_for("train"))

    preview = None

    # Jika ada hasil testing terakhir, tampilkan preview-nya di GET
    last_test = APP_STATE.get("last_test_path")
    if request.method == "GET" and last_test and Path(last_test).exists():
        try:
            df_prev = pd.read_excel(last_test, engine="openpyxl") if str(last_test).lower().endswith((".xls", ".xlsx")) else pd.read_csv(last_test)
            preview = df_prev.head(15).to_html(classes="table table-sm table-striped", index=False)
            # pastikan APP_STATE test counts sudah set (jika belum)
            if "pred" in df_prev.columns:
                counts = df_prev["pred"].value_counts().to_dict()
                APP_STATE["test_counts"] = {
                    "positif": int(counts.get("positif", 0)),
                    "negatif": int(counts.get("negatif", 0)),
                    "netral":  int(counts.get("netral", 0)),
                }
                APP_STATE["n_test"] = int(len(df_prev))
        except Exception as e:
            print("[WARN] gagal muat last_test_path preview:", e)

    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            flash("Pilih file CSV testing (kolom: teks).", "warning")
            return redirect(url_for("test_page"))
        if not allowed_file(f.filename):
            flash("Hanya mendukung file .csv", "danger")
            return redirect(url_for("test_page"))

        test_path = UPLOAD_DIR / "test.csv"
        f.save(test_path)

        try:
            df = pd.read_csv(test_path)
        except Exception as e:
            flash(f"Gagal membaca CSV: {e}", "danger")
            return redirect(url_for("test_page"))

        if "teks" not in df.columns:
            flash("CSV testing harus memiliki kolom 'teks'.", "danger")
            return redirect(url_for("test_page"))

        model = joblib.load(model_path)
        df["clean"] = df["teks"].astype(str).apply(indo_clean_final)
        df["pred"] = model.predict(df["clean"])

        counts = df["pred"].value_counts().to_dict()
        APP_STATE["test_counts"] = {
            "positif": int(counts.get("positif", 0)),
            "negatif": int(counts.get("negatif", 0)),
            "netral":  int(counts.get("netral", 0)),
        }
        APP_STATE["n_test"] = int(len(df))

        out_path = OUTPUTS_DIR / f"testing_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df[["teks", "pred"]].to_excel(out_path, index=False)
        APP_STATE["last_test_path"] = str(out_path)

        flash("Testing selesai. Anda dapat mengunduh hasil prediksi.", "success")
        preview = df.head(15).to_html(classes="table table-sm table-striped", index=False)

    return render_template("test.html", preview_html=preview, state=APP_STATE)


@app.route("/download/testing")
@login_required
def download_testing():
    path = APP_STATE.get("last_test_path")
    if not path or not Path(path).exists():
        flash("Belum ada hasil testing untuk diunduh.", "warning")
        return redirect(url_for("test_page"))
    return send_file(path, as_attachment=True)

# ==== CEK KALIMAT ====
@app.route("/cek", methods=["GET", "POST"])
@login_required
def cek_kalimat():
    model_path = MODELS_DIR / "nb_tfidf_pipeline.joblib"
    if not model_path.exists():
        flash("Model belum dilatih. Silakan lakukan training terlebih dahulu.", "warning")
        return redirect(url_for("train"))

    model = joblib.load(model_path)
    hasil_pred, teks_asli, teks_bersih = None, "", ""
    proba_dict = None

    # Threshold anti-bias
    NEUTRAL_MARGIN = 0.12   # selisih dua proba teratas
    MIN_CONFIDENCE = 0.55   # ambang keyakinan minimal

    if request.method == "POST":
        teks_asli = request.form.get("kalimat", "")
        if teks_asli.strip() == "":
            flash("Masukkan kalimat terlebih dahulu.", "warning")
            return redirect(url_for("cek_kalimat"))

        # cleansing final
        teks_bersih = indo_clean_final(teks_asli)

        # 1) Booster: jika mengandung kata positif khas data train
        words = set(teks_bersih.split())
        if len(words & GLOBAL_POSITIVE_HINT) > 0:
            hasil_pred = "positif"
        else:
            # 2) Prediksi probabilitas
            proba = model.predict_proba([teks_bersih])[0]
            classes = list(model.classes_)
            proba_dict = {c: float(p) for c, p in zip(classes, proba)}

            # ambil dua nilai teratas
            idx_top = int(proba.argmax())
            val_top = float(proba[idx_top])
            lbl_top = classes[idx_top]
            val_second = float(sorted(proba)[-2]) if len(proba) >= 2 else 0.0

            # 3) Aturan netral bila ragu
            if (val_top < MIN_CONFIDENCE) or ((val_top - val_second) < NEUTRAL_MARGIN):
                hasil_pred = "netral"
            else:
                hasil_pred = lbl_top

    return render_template(
        "cek_kalimat.html",
        hasil_pred=hasil_pred,
        teks_asli=teks_asli,
        teks_bersih=teks_bersih,
        proba=proba_dict
    )

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    print("URL Map:\n", app.url_map)  # untuk debug route
    app.run(debug=True)