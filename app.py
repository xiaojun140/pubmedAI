import os
import re
import time
import base64
import sqlite3
import threading
import hashlib
import hmac
from datetime import datetime
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st

# =========================
# Global auth DB (shared)
# =========================
AUTH_DB_FILE = "users.db"

# =========================
# Per-user DB directory
# =========================
DATA_DIR = "user_dbs"
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# Locks
# =========================
_AUTH_LOCK = threading.RLock()
_USER_DB_LOCKS: dict[str, threading.RLock] = {}  # user_id -> lock


def _get_user_lock(user_id: str) -> threading.RLock:
    if user_id not in _USER_DB_LOCKS:
        _USER_DB_LOCKS[user_id] = threading.RLock()
    return _USER_DB_LOCKS[user_id]


# ======================================================
# Helpers: password hashing (PBKDF2)
# ======================================================
def _pbkdf2_hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = _pbkdf2_hash_password(password, salt)
    return base64.b64encode(salt + dk).decode("utf-8")


def verify_password(password: str, stored: str) -> bool:
    raw = base64.b64decode(stored.encode("utf-8"))
    salt, dk = raw[:16], raw[16:]
    dk2 = _pbkdf2_hash_password(password, salt)
    return hmac.compare_digest(dk, dk2)


def safe_filename(s: str) -> str:
    # strictly reduce risk of path traversal
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)


# ======================================================
# DB connections
# ======================================================
def auth_db_connect():
    conn = sqlite3.connect(AUTH_DB_FILE, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=8000;")
    return conn


def user_db_path(user_id: str) -> str:
    # per-user file
    return os.path.join(DATA_DIR, f"db_{safe_filename(user_id)}.db")


def user_db_connect(user_id: str):
    conn = sqlite3.connect(user_db_path(user_id), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=8000;")
    return conn


def auth_db_exec(sql: str, params=()):
    with _AUTH_LOCK:
        conn = auth_db_connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
            return cur
        finally:
            conn.close()


def auth_db_one(sql: str, params=()):
    with _AUTH_LOCK:
        conn = auth_db_connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            return cur.fetchone()
        finally:
            conn.close()


def user_db_exec(user_id: str, sql: str, params=()):
    lock = _get_user_lock(user_id)
    with lock:
        conn = user_db_connect(user_id)
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
            return cur
        finally:
            conn.close()


def user_db_df(user_id: str, sql: str, params=()):
    lock = _get_user_lock(user_id)
    with lock:
        conn = user_db_connect(user_id)
        try:
            return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()


def user_db_one(user_id: str, sql: str, params=()):
    lock = _get_user_lock(user_id)
    with lock:
        conn = user_db_connect(user_id)
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            return cur.fetchone()
        finally:
            conn.close()


# ======================================================
# Init DBs
# ======================================================
def init_auth_db():
    with _AUTH_LOCK:
        conn = auth_db_connect()
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()


def init_user_db(user_id: str):
    lock = _get_user_lock(user_id)
    with lock:
        conn = user_db_connect(user_id)
        try:
            c = conn.cursor()

            c.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    pmid TEXT PRIMARY KEY,
                    title TEXT,
                    journal TEXT,
                    year INTEGER,
                    abstract TEXT,
                    doi TEXT,
                    pmcid TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS favorites (
                    pmid TEXT PRIMARY KEY,
                    FOREIGN KEY (pmid) REFERENCES articles(pmid)
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    idx INTEGER PRIMARY KEY,
                    pmid TEXT UNIQUE,
                    FOREIGN KEY (pmid) REFERENCES articles(pmid)
                )
            """)

            # API Key: NOT stored (non-reversible requirement)
            c.execute("""
                CREATE TABLE IF NOT EXISTS ai_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    base_url TEXT,
                    model TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    system_prompt TEXT,
                    api_key_hash TEXT  -- optional: only indicates "set" status, cannot recover key
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS ai_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    source TEXT,
                    pmids TEXT,
                    topic_hint TEXT,
                    base_url TEXT,
                    model TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    system_prompt TEXT,
                    user_prompt TEXT,
                    output TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS ai_chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    chat_type TEXT,
                    base_url TEXT,
                    model TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    system_prompt TEXT,
                    user_input TEXT,
                    assistant_output TEXT
                )
            """)

            # Multi-turn review chat sessions
            c.execute("""
                CREATE TABLE IF NOT EXISTS review_chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    updated_at TEXT,
                    pmids TEXT,
                    topic_hint TEXT,
                    context TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS review_chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    created_at TEXT,
                    role TEXT,
                    content TEXT,
                    FOREIGN KEY(session_id) REFERENCES review_chat_sessions(id)
                )
            """)

            conn.commit()
        finally:
            conn.close()


# ======================================================
# User Auth
# ======================================================
def create_user(username: str, password: str) -> str:
    username = username.strip()
    if not username:
        raise ValueError("用户名不能为空")
    if len(password) < 6:
        raise ValueError("密码至少 6 位")

    user_id = hashlib.sha256((username + "|" + str(time.time())).encode("utf-8")).hexdigest()[:16]
    pw_hash = hash_password(password)
    auth_db_exec(
        "INSERT INTO users(user_id, username, password_hash, created_at) VALUES(?,?,?,?)",
        (user_id, username, pw_hash, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    # create per-user db file now
    init_user_db(user_id)
    return user_id


def login_user(username: str, password: str) -> str | None:
    row = auth_db_one("SELECT user_id, password_hash FROM users WHERE username=?", (username.strip(),))
    if not row:
        return None
    user_id, pw_hash = row
    if verify_password(password, pw_hash):
        init_user_db(user_id)
        return user_id
    return None


def get_username(user_id: str) -> str:
    row = auth_db_one("SELECT username FROM users WHERE user_id=?", (user_id,))
    return row[0] if row else "Unknown"


# ======================================================
# AI settings (per-user DB, API Key not stored)
# ======================================================
def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_ai_settings(user_id: str):
    row = user_db_one(
        user_id,
        "SELECT base_url, model, temperature, max_tokens, system_prompt, api_key_hash FROM ai_settings WHERE id=1"
    )
    if not row:
        return {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 1500,
            "system_prompt": "",
            "api_key_hash": "",
        }
    return {
        "base_url": row[0] or "https://api.openai.com/v1",
        "model": row[1] or "gpt-4o-mini",
        "temperature": float(row[2]) if row[2] is not None else 0.3,
        "max_tokens": int(row[3]) if row[3] is not None else 1500,
        "system_prompt": row[4] or "",
        "api_key_hash": row[5] or "",
    }


def save_ai_settings(user_id: str, base_url, model, temperature, max_tokens, system_prompt, api_key_plain: str | None):
    """
    API Key is NOT stored. Only store hash (optional) to show "已设置" status.
    """
    api_key_hash = _sha256_hex(api_key_plain) if api_key_plain else (load_ai_settings(user_id).get("api_key_hash") or "")
    user_db_exec(
        user_id,
        """
        INSERT INTO ai_settings(id, base_url, model, temperature, max_tokens, system_prompt, api_key_hash)
        VALUES(1,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            base_url=excluded.base_url,
            model=excluded.model,
            temperature=excluded.temperature,
            max_tokens=excluded.max_tokens,
            system_prompt=excluded.system_prompt,
            api_key_hash=excluded.api_key_hash
        """,
        ((base_url or "").rstrip("/"), model, float(temperature), int(max_tokens), system_prompt or "", api_key_hash),
    )


# ======================================================
# Search cache (per-user DB)
# ======================================================
def clear_search_cache(user_id: str):
    user_db_exec(user_id, "DELETE FROM search_cache")


def save_search_results_to_db(user_id: str, articles: list[dict]):
    lock = _get_user_lock(user_id)
    with lock:
        conn = user_db_connect(user_id)
        try:
            c = conn.cursor()
            c.execute("DELETE FROM search_cache")

            for i, article in enumerate(articles):
                pmid = article.get("pmid")
                if not pmid:
                    continue

                year_val = None
                y = article.get("year")
                if y and str(y).isdigit():
                    year_val = int(y)

                c.execute("""
                    INSERT INTO articles(pmid,title,journal,year,abstract,doi,pmcid)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(pmid) DO UPDATE SET
                        title=excluded.title,
                        journal=excluded.journal,
                        year=excluded.year,
                        abstract=excluded.abstract,
                        doi=excluded.doi,
                        pmcid=excluded.pmcid
                """, (pmid, article.get("title"), article.get("journal"), year_val,
                      article.get("abstract"), article.get("doi"), article.get("pmcid")))

                c.execute("INSERT OR REPLACE INTO search_cache(idx, pmid) VALUES(?,?)", (i, pmid))

            conn.commit()
        finally:
            conn.close()


def get_search_total_count(user_id: str) -> int:
    row = user_db_one(user_id, "SELECT COUNT(*) FROM search_cache")
    return int(row[0]) if row else 0


def load_search_page(user_id: str, page: int, page_size: int) -> pd.DataFrame:
    offset = page * page_size
    return user_db_df(user_id, """
        SELECT a.*
        FROM search_cache s
        JOIN articles a ON a.pmid = s.pmid
        ORDER BY s.idx
        LIMIT ? OFFSET ?
    """, (page_size, offset))


def load_articles_by_pmids(user_id: str, pmids: list[str]) -> pd.DataFrame:
    if not pmids:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(pmids))
    df = user_db_df(user_id, f"SELECT * FROM articles WHERE pmid IN ({placeholders})", tuple(pmids))

    order_map = {pmid: i for i, pmid in enumerate(pmids)}
    if not df.empty:
        df["__ord"] = df["pmid"].map(order_map)
        df = df.sort_values("__ord").drop(columns="__ord")
    return df


# ======================================================
# Favorites (per-user DB)
# ======================================================
def add_favorite(user_id: str, article: dict):
    lock = _get_user_lock(user_id)
    with lock:
        conn = user_db_connect(user_id)
        try:
            c = conn.cursor()

            year_val = None
            y = article.get("year")
            if y and str(y).isdigit():
               
