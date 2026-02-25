import os
import base64
import time
import sqlite3
import threading
import hashlib
import hmac
from datetime import datetime
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st

# ========= Optional dependency for encryption =========
try:
    from cryptography.fernet import Fernet, InvalidToken
except Exception:
    Fernet = None
    InvalidToken = Exception

DB_FILE = "articles.db"

# ========= DB lock (in-process) =========
_DB_LOCK = threading.RLock()


# ======================================================
# Security: API key encryption (Fernet)
# ======================================================
def _get_master_fernet() -> "Fernet":
    """
    Use env var APP_MASTER_KEY as Fernet key.
    """
    if Fernet is None:
        raise RuntimeError("缺少依赖 cryptography。请先安装：pip install cryptography")

    k = os.environ.get("APP_MASTER_KEY", "").strip()
    if not k:
        raise RuntimeError(
            "未设置环境变量 APP_MASTER_KEY（用于加密 API Key）。\n"
            "请先生成：python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\" \n"
            "并设置：APP_MASTER_KEY=<生成值>"
        )
    try:
        return Fernet(k.encode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"APP_MASTER_KEY 非法：{e}")


def encrypt_secret(plain: str) -> str:
    if not plain:
        return ""
    f = _get_master_fernet()
    token = f.encrypt(plain.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_secret(cipher: str) -> str:
    if not cipher:
        return ""
    f = _get_master_fernet()
    try:
        data = f.decrypt(cipher.encode("utf-8"))
        return data.decode("utf-8")
    except InvalidToken:
        # If key changed, or corrupted, treat as empty but don't crash whole app
        return ""


# ======================================================
# Security: user auth (local, PBKDF2)
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


# ======================================================
# DB helpers (locking + WAL + busy_timeout)
# ======================================================
def db_connect():
    """
    One connection per operation; SQLite is file-based; this is okay for Streamlit.
    WAL + busy_timeout reduces 'database is locked' errors.
    """
    conn = sqlite3.connect(DB_FILE, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=8000;")
    return conn


def db_exec(sql: str, params=()):
    with _DB_LOCK:
        conn = db_connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
            return cur
        finally:
            conn.close()


def db_query_df(sql: str, params=()):
    with _DB_LOCK:
        conn = db_connect()
        try:
            df = pd.read_sql_query(sql, conn, params=params)
            return df
        finally:
            conn.close()


def db_query_one(sql: str, params=()):
    with _DB_LOCK:
        conn = db_connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            return cur.fetchone()
        finally:
            conn.close()


def _table_has_column(table: str, col: str) -> bool:
    row = db_query_df(f"PRAGMA table_info({table})")
    if row.empty:
        return False
    return col in row["name"].tolist()


# ======================================================
# Init / migrations
# ======================================================
def init_db():
    with _DB_LOCK:
        conn = db_connect()
        try:
            c = conn.cursor()

            # --- users ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    created_at TEXT
                )
            """)

            # --- articles (global, shared) ---
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

            # --- favorites (per user) ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS favorites (
                    user_id TEXT,
                    pmid TEXT,
                    PRIMARY KEY (user_id, pmid),
                    FOREIGN KEY (pmid) REFERENCES articles(pmid)
                )
            """)

            # --- search_cache (per user, per session) ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    user_id TEXT,
                    idx INTEGER,
                    pmid TEXT,
                    PRIMARY KEY (user_id, idx),
                    UNIQUE (user_id, pmid),
                    FOREIGN KEY (pmid) REFERENCES articles(pmid)
                )
            """)

            # --- AI settings (per user) ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS ai_settings (
                    user_id TEXT PRIMARY KEY,
                    base_url TEXT,
                    api_key_enc TEXT,
                    model TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    system_prompt TEXT
                )
            """)

            # --- AI reviews history (per user) ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS ai_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
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

            # --- AI chat logs (per user) ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS ai_chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
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

            # --- NEW: multi-turn review chat sessions ---
            c.execute("""
                CREATE TABLE IF NOT EXISTS review_chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
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
                    user_id TEXT,
                    session_id INTEGER,
                    created_at TEXT,
                    role TEXT,
                    content TEXT,
                    FOREIGN KEY (session_id) REFERENCES review_chat_sessions(id)
                )
            """)

            conn.commit()
        finally:
            conn.close()

    # Best-effort migration from old single-user schema (if present)
    # - If old favorites table had only pmid (PRIMARY KEY), move to new favorites with user_id='default'
    # - If old search_cache had (idx, pmid), move to new search_cache with user_id='default'
    try:
        migrate_legacy_to_multiuser()
    except Exception:
        # Never block app start on migration errors
        pass


def migrate_legacy_to_multiuser():
    """
    If the DB was created by your old version, attempt to migrate data.
    This function is idempotent-ish and best-effort.
    """
    with _DB_LOCK:
        conn = db_connect()
        try:
            c = conn.cursor()

            # detect legacy favorites schema: (pmid TEXT PRIMARY KEY)
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='favorites'")
            if not c.fetchone():
                return

            # Check if favorites has user_id column; if not, it is legacy
            c.execute("PRAGMA table_info(favorites)")
            cols = [r[1] for r in c.fetchall()]
            if "user_id" in cols:
                return  # already new

            # Legacy detected. Rename and rebuild.
            c.execute("ALTER TABLE favorites RENAME TO favorites_legacy")
            c.execute("""
                CREATE TABLE IF NOT EXISTS favorites (
                    user_id TEXT,
                    pmid TEXT,
                    PRIMARY KEY (user_id, pmid),
                    FOREIGN KEY (pmid) REFERENCES articles(pmid)
                )
            """)
            c.execute("INSERT OR IGNORE INTO favorites(user_id, pmid) SELECT 'default', pmid FROM favorites_legacy")
            c.execute("DROP TABLE favorites_legacy")

            # search_cache legacy?
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='search_cache'")
            if c.fetchone():
                c.execute("PRAGMA table_info(search_cache)")
                cols2 = [r[1] for r in c.fetchall()]
                if "user_id" not in cols2:
                    c.execute("ALTER TABLE search_cache RENAME TO search_cache_legacy")
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS search_cache (
                            user_id TEXT,
                            idx INTEGER,
                            pmid TEXT,
                            PRIMARY KEY (user_id, idx),
                            UNIQUE (user_id, pmid),
                            FOREIGN KEY (pmid) REFERENCES articles(pmid)
                        )
                    """)
                    c.execute("""
                        INSERT OR IGNORE INTO search_cache(user_id, idx, pmid)
                        SELECT 'default', idx, pmid FROM search_cache_legacy
                    """)
                    c.execute("DROP TABLE search_cache_legacy")

            # ai_settings legacy? (single row id=1)
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_settings'")
            if c.fetchone():
                c.execute("PRAGMA table_info(ai_settings)")
                cols3 = [r[1] for r in c.fetchall()]
                if "user_id" not in cols3 and "id" in cols3:
                    # legacy table: id=1 row
                    row = conn.execute("SELECT base_url, api_key, model, temperature, max_tokens, system_prompt FROM ai_settings WHERE id=1").fetchone()
                    c.execute("ALTER TABLE ai_settings RENAME TO ai_settings_legacy")
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS ai_settings (
                            user_id TEXT PRIMARY KEY,
                            base_url TEXT,
                            api_key_enc TEXT,
                            model TEXT,
                            temperature REAL,
                            max_tokens INTEGER,
                            system_prompt TEXT
                        )
                    """)
                    if row:
                        base_url, api_key, model, temperature, max_tokens, system_prompt = row
                        api_key_enc = encrypt_secret(api_key or "")
                        c.execute("""
                            INSERT OR REPLACE INTO ai_settings(user_id, base_url, api_key_enc, model, temperature, max_tokens, system_prompt)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, ("default", base_url or "https://api.openai.com/v1", api_key_enc, model or "gpt-4o-mini",
                              float(temperature) if temperature is not None else 0.3,
                              int(max_tokens) if max_tokens is not None else 1500,
                              system_prompt or ""))
                    c.execute("DROP TABLE ai_settings_legacy")

            conn.commit()
        finally:
            conn.close()


# ======================================================
# User auth DB operations
# ======================================================
def create_user(username: str, password: str) -> str:
    username = username.strip()
    if not username:
        raise ValueError("用户名不能为空")
    if len(password) < 6:
        raise ValueError("密码至少 6 位")

    user_id = hashlib.sha256((username + "|" + str(time.time())).encode("utf-8")).hexdigest()[:16]
    pw_hash = hash_password(password)
    db_exec(
        "INSERT INTO users(user_id, username, password_hash, created_at) VALUES(?,?,?,?)",
        (user_id, username, pw_hash, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    return user_id


def login_user(username: str, password: str) -> str | None:
    row = db_query_one("SELECT user_id, password_hash FROM users WHERE username = ?", (username.strip(),))
    if not row:
        return None
    user_id, pw_hash = row
    if verify_password(password, pw_hash):
        return user_id
    return None


def get_username(user_id: str) -> str:
    row = db_query_one("SELECT username FROM users WHERE user_id=?", (user_id,))
    return row[0] if row else "Unknown"


# ======================================================
# AI settings (per user) read/write
# ======================================================
def load_ai_settings(user_id: str):
    row = db_query_one(
        "SELECT base_url, api_key_enc, model, temperature, max_tokens, system_prompt FROM ai_settings WHERE user_id=?",
        (user_id,),
    )
    if not row:
        return {
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 1500,
            "system_prompt": "",
        }
    return {
        "base_url": row[0] or "https://api.openai.com/v1",
        "api_key": decrypt_secret(row[1] or ""),
        "model": row[2] or "gpt-4o-mini",
        "temperature": float(row[3]) if row[3] is not None else 0.3,
        "max_tokens": int(row[4]) if row[4] is not None else 1500,
        "system_prompt": row[5] or "",
    }


def save_ai_settings(user_id: str, base_url, api_key, model, temperature, max_tokens, system_prompt):
    api_key_enc = encrypt_secret(api_key or "")
    db_exec(
        """
        INSERT INTO ai_settings(user_id, base_url, api_key_enc, model, temperature, max_tokens, system_prompt)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            base_url=excluded.base_url,
            api_key_enc=excluded.api_key_enc,
            model=excluded.model,
            temperature=excluded.temperature,
            max_tokens=excluded.max_tokens,
            system_prompt=excluded.system_prompt
        """,
        (user_id, (base_url or "").rstrip("/"), api_key_enc, model, float(temperature), int(max_tokens), system_prompt or ""),
    )


# ======================================================
# Articles / Favorites / Search cache (per user)
# ======================================================
def clear_search_cache(user_id: str):
    with _DB_LOCK:
        conn = db_connect()
        try:
            c = conn.cursor()
            # delete cached-only articles not favorited by this user AND not favorited by anyone? safer: keep global articles.
            # We'll only clear search_cache rows; do not delete articles table rows (shared).
            c.execute("DELETE FROM search_cache WHERE user_id=?", (user_id,))
            conn.commit()
        finally:
            conn.close()


def save_search_results_to_db(user_id: str, articles: list[dict]):
    with _DB_LOCK:
        conn = db_connect()
        try:
            c = conn.cursor()
            c.execute("DELETE FROM search_cache WHERE user_id=?", (user_id,))

            for i, article in enumerate(articles):
                pmid = article.get("pmid")
                if not pmid:
                    continue

                year_val = None
                y = article.get("year")
                if y and str(y).isdigit():
                    year_val = int(y)

                c.execute(
                    """
                    INSERT INTO articles(pmid,title,journal,year,abstract,doi,pmcid)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(pmid) DO UPDATE SET
                        title=excluded.title,
                        journal=excluded.journal,
                        year=excluded.year,
                        abstract=excluded.abstract,
                        doi=excluded.doi,
                        pmcid=excluded.pmcid
                    """,
                    (
                        pmid,
                        article.get("title"),
                        article.get("journal"),
                        year_val,
                        article.get("abstract"),
                        article.get("doi"),
                        article.get("pmcid"),
                    ),
                )

                c.execute(
                    "INSERT OR REPLACE INTO search_cache(user_id, idx, pmid) VALUES(?,?,?)",
                    (user_id, i, pmid),
                )

            conn.commit()
        finally:
            conn.close()


def get_search_total_count(user_id: str) -> int:
    row = db_query_one("SELECT COUNT(*) FROM search_cache WHERE user_id=?", (user_id,))
    return int(row[0]) if row else 0


def load_search_page(user_id: str, page: int, page_size: int) -> pd.DataFrame:
    offset = page * page_size
    return db_query_df(
        """
        SELECT a.*
        FROM search_cache s
        JOIN articles a ON a.pmid = s.pmid
        WHERE s.user_id=?
        ORDER BY s.idx
        LIMIT ? OFFSET ?
        """,
        (user_id, page_size, offset),
    )


def load_articles_by_pmids(pmids: list[str]) -> pd.DataFrame:
    if not pmids:
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(pmids))
    df = db_query_df(f"SELECT * FROM articles WHERE pmid IN ({placeholders})", tuple(pmids))

    order_map = {pmid: i for i, pmid in enumerate(pmids)}
    if not df.empty:
        df["__ord"] = df["pmid"].map(order_map)
        df = df.sort_values("__ord").drop(columns="__ord")
    return df


def add_favorite(user_id: str, article: dict):
    with _DB_LOCK:
        conn = db_connect()
        try:
            c = conn.cursor()

            year_val = None
            y = article.get("year")
            if y and str(y).isdigit():
                year_val = int(y)

            c.execute(
                """
                INSERT INTO articles(pmid,title,journal,year,abstract,doi,pmcid)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(pmid) DO UPDATE SET
                    title=excluded.title,
                    journal=excluded.journal,
                    year=excluded.year,
                    abstract=excluded.abstract,
                    doi=excluded.doi,
                    pmcid=excluded.pmcid
                """,
                (
                    article.get("pmid"),
                    article.get("title"),
                    article.get("journal"),
                    year_val,
                    article.get("abstract"),
                    article.get("doi"),
                    article.get("pmcid"),
                ),
            )
            c.execute("INSERT OR IGNORE INTO favorites(user_id, pmid) VALUES(?,?)", (user_id, article.get("pmid")))
            conn.commit()
        finally:
            conn.close()


def remove_favorite(user_id: str, pmid: str):
    db_exec("DELETE FROM favorites WHERE user_id=? AND pmid=?", (user_id, pmid))


def clear_favorites(user_id: str):
    db_exec("DELETE FROM favorites WHERE user_id=?", (user_id,))


def load_favorites(user_id: str) -> pd.DataFrame:
    return db_query_df(
        """
        SELECT a.*
        FROM articles a
        JOIN favorites f ON a.pmid = f.pmid
        WHERE f.user_id=?
        """,
        (user_id,),
    )


def load_favorite_pmids(user_id: str) -> list[str]:
    df = db_query_df("SELECT pmid FROM favorites WHERE user_id=?", (user_id,))
    return df["pmid"].tolist() if not df.empty else []


# ======================================================
# Export
# ======================================================
def generate_ris(df: pd.DataFrame) -> str:
    ris_content = ""
    for _, a in df.iterrows():
        ris_content += "TY  - JOUR\n"
        ris_content += f"TI  - {a.get('title', '')}\n"
        if a.get("doi"):
            ris_content += f"DO  - {a['doi']}\n"
        if a.get("year"):
            ris_content += f"PY  - {a['year']}\n"
        ris_content += "ER  - \n\n"
    return ris_content


def trigger_frontend_download(filename: str, mime: str, data_bytes: bytes):
    if "dl_nonce" not in st.session_state:
        st.session_state["dl_nonce"] = 0
    st.session_state["dl_nonce"] += 1

    nonce = f"{st.session_state['dl_nonce']}_{int(time.time() * 1000)}"
    anchor_id = f"dl_{nonce}"

    b64 = base64.b64encode(data_bytes).decode("utf-8")
    html = f"""
    <html>
      <body>
        <a id="{anchor_id}" download="{filename}" href="data:{mime};base64,{b64}"></a>
        <script>
          (function() {{
            const a = document.getElementById("{anchor_id}");
            setTimeout(() => {{ a.click(); }}, 50);
          }})();
        </script>
      </body>
    </html>
    """
    st.components.v1.html(html, height=0, width=0)


def show_dialog(title, msg, state_key_to_clear=None):
    try:
        @st.dialog(title)
        def _dlg():
            st.warning(msg)
            if st.button("知道了"):
                if state_key_to_clear:
                    st.session_state[state_key_to_clear] = None
                st.rerun()
        _dlg()
    except Exception:
        st.warning(msg)
        if state_key_to_clear:
            st.session_state[state_key_to_clear] = None


# ======================================================
# External: citation count
# ======================================================
@st.cache_data(show_spinner=False)
def get_citation_count(pmid):
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/PMID:{pmid}?fields=citationCount"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json().get("citationCount", 0)
    except Exception:
        pass
    return None


def highlight_keywords(text, keywords):
    if not text:
        return ""
    highlighted = text
    for word in (keywords or "").split():
        w = word.strip()
        if w:
            highlighted = highlighted.replace(w, f"<span style='background-color:yellow'>{w}</span>")
    return highlighted


# ======================================================
# PubMed search
# ======================================================
def search_pubmed(query, year_from, year_to, article_type, retmax=20):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    query_full = query

    if year_from and year_to:
        query_full += f" AND {year_from}:{year_to}[dp]"
    if article_type != "All":
        query_full += f" AND {article_type}[pt]"

    search = requests.get(
        base + "esearch.fcgi",
        params={"db": "pubmed", "term": query_full, "retmax": retmax, "sort": "pub date", "retmode": "json"},
        timeout=15,
    ).json()

    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    fetch = requests.get(
        base + "efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        timeout=30,
    )

    root = ET.fromstring(fetch.content)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        articles.append(
            {
                "pmid": article.findtext(".//PMID"),
                "title": article.findtext(".//ArticleTitle"),
                "journal": article.findtext(".//Journal/Title"),
                "year": article.findtext(".//PubDate/Year"),
                "abstract": " ".join([a.text for a in article.findall(".//AbstractText") if a.text]),
                "doi": next((a.text for a in article.findall(".//ArticleId") if a.attrib.get("IdType") == "doi"), None),
                "pmcid": next((a.text for a in article.findall(".//ArticleId") if a.attrib.get("IdType") == "pmc"), None),
            }
        )
    return articles


# ======================================================
# OpenAI-compatible Chat Completions
# ======================================================
def call_chat_completions(base_url, api_key, model, temperature, max_tokens, system_prompt, messages):
    """
    messages: list of {"role": "...", "content": "..."}.
    Will prepend system prompt.
    """
    base_url = (base_url or "").rstrip("/")
    url = f"{base_url}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    full_messages = [{"role": "system", "content": system_prompt or ""}] + messages

    payload = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": full_messages,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"接口返回 {r.status_code}: {r.text[:500]}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"解析返回失败：{data}")


# ======================================================
# Review context builder
# ======================================================
def build_ai_context(df: pd.DataFrame, max_chars: int = 28000) -> str:
    parts, used = [], 0
    for _, r in df.iterrows():
        pmid = str(r.get("pmid", "") or "")
        title = (r.get("title", "") or "").strip()
        abstract = (r.get("abstract", "") or "").strip()
        journal = (r.get("journal", "") or "").strip()
        year = r.get("year", "")

        block = f"[PMID:{pmid}]\nTitle: {title}\nJournal: {journal}\nYear: {year}\nAbstract: {abstract}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n---\n".join(parts)


def review_system_prompt(custom_system_prompt: str = "") -> str:
    system_default = (
        "你是一名严谨的医学/生命科学综述写作助手。"
        "你必须只依据用户提供的参考文献信息（题目与摘要）进行归纳，避免臆测。"
        "输出为中文。"
        "非常重要：每一句话末尾必须用括号标注 PMID 作为来源，格式严格为 (PMID:12345678) 或 (PMID:123; PMID:456)。"
        "如果一句话综合多篇文献，列出多个 PMID。"
        "不要在句中插入 PMID，只能句末标注。"
        "不要输出参考文献列表（用户将用 EndNote 处理）。"
        "如果用户追问，请继续基于同一批文献与历史对话进行回答，仍然遵守每句句末 PMID 标注规则。"
    )
    if custom_system_prompt.strip():
        return system_default + "\n\n" + custom_system_prompt.strip()
    return system_default


def build_review_first_turn(topic_hint: str, df: pd.DataFrame, user_extra: str = "") -> str:
    context = build_ai_context(df)

    topic_value = topic_hint.strip() or "由文献内容自动归纳主题"
    default_requirement = (
        "1 建议结构：背景与问题  关键机制或证据  临床、应用或研究进展 局限性 未来方向\n"
        "2 尽量做到归纳对比，避免逐篇复述。\n"
        "3 任何一句话都必须在句末标注 PMID。\n"
        "4 禁止编造，若文献中无信息支撑，请使用谨慎表达，并仍标注来源 PMID。\n"
        "5 篇幅：约 800~1500 字，可根据文献数量适当调整。"
    )
    extra_requirement = user_extra.strip() or default_requirement

    user_prompt = f"""
请基于下面提供的文献题目与摘要，围绕主题生成一篇结构清晰的综述。

【主题/方向】
{topic_value}

【写作要求】
{extra_requirement}

【文献数据】
{context}
""".strip()
    return user_prompt


# ======================================================
# Multi-turn review chat: DB operations
# ======================================================
def create_review_chat_session(user_id: str, pmids: list[str], topic_hint: str, context: str) -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pmids_s = ",".join([str(p) for p in pmids])
    with _DB_LOCK:
        conn = db_connect()
        try:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO review_chat_sessions(user_id, created_at, updated_at, pmids, topic_hint, context)
                VALUES(?,?,?,?,?,?)
                """,
                (user_id, now, now, pmids_s, topic_hint or "", context),
            )
            conn.commit()
            return int(c.lastrowid)
        finally:
            conn.close()


def touch_review_chat_session(user_id: str, session_id: int):
    db_exec(
        "UPDATE review_chat_sessions SET updated_at=? WHERE user_id=? AND id=?",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, session_id),
    )


def add_review_chat_message(user_id: str, session_id: int, role: str, content: str):
    db_exec(
        """
        INSERT INTO review_chat_messages(user_id, session_id, created_at, role, content)
        VALUES(?,?,?,?,?)
        """,
        (user_id, session_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), role, content),
    )
    touch_review_chat_session(user_id, session_id)


def load_review_chat_session(user_id: str, session_id: int):
    df = db_query_df(
        "SELECT * FROM review_chat_sessions WHERE user_id=? AND id=?",
        (user_id, session_id),
    )
    return df.iloc[0].to_dict() if not df.empty else None


def list_review_chat_sessions(user_id: str, limit: int = 50) -> pd.DataFrame:
    return db_query_df(
        """
        SELECT id, created_at, updated_at, topic_hint, substr(pmids,1,80) AS pmids_preview
        FROM review_chat_sessions
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit),
    )


def load_review_chat_messages(user_id: str, session_id: int, limit: int = 200) -> list[dict]:
    df = db_query_df(
        """
        SELECT role, content, created_at
        FROM review_chat_messages
        WHERE user_id=? AND session_id=?
        ORDER BY id ASC
        LIMIT ?
        """,
        (user_id, session_id, limit),
    )
    return df.to_dict(orient="records") if not df.empty else []


# ======================================================
# Legacy AI review history (kept)
# ======================================================
def save_ai_review_to_db(user_id: str, source, pmids, topic_hint, base_url, model, temperature, max_tokens, system_prompt, user_prompt, output):
    db_exec(
        """
        INSERT INTO ai_reviews(
            user_id, created_at, source, pmids, topic_hint,
            base_url, model, temperature, max_tokens,
            system_prompt, user_prompt, output
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            user_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source,
            pmids,
            topic_hint,
            base_url,
            model,
            float(temperature),
            int(max_tokens),
            system_prompt,
            user_prompt,
            output,
        ),
    )


def list_ai_reviews(user_id: str, limit=50) -> pd.DataFrame:
    return db_query_df(
        """
        SELECT id, created_at, source, topic_hint, model
        FROM ai_reviews
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit),
    )


def load_ai_review(user_id: str, review_id: int):
    df = db_query_df(
        "SELECT * FROM ai_reviews WHERE user_id=? AND id=?",
        (user_id, review_id),
    )
    return df.iloc[0].to_dict() if not df.empty else None


# ======================================================
# AI chat logs for PubMed strategy (kept)
# ======================================================
def save_chat_log(user_id: str, chat_type, base_url, model, temperature, max_tokens, system_prompt, user_input, assistant_output):
    db_exec(
        """
        INSERT INTO ai_chat_logs(
            user_id, created_at, chat_type, base_url, model, temperature, max_tokens,
            system_prompt, user_input, assistant_output
        )
        VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (
            user_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            chat_type,
            base_url,
            model,
            float(temperature),
            int(max_tokens),
            system_prompt,
            user_input,
            assistant_output,
        ),
    )


def list_chat_logs(user_id: str, chat_type, limit=50) -> pd.DataFrame:
    return db_query_df(
        """
        SELECT id, created_at, model, substr(user_input,1,60) as user_input
        FROM ai_chat_logs
        WHERE user_id=? AND chat_type=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, chat_type, limit),
    )


def load_chat_log(user_id: str, chat_id: int):
    df = db_query_df(
        "SELECT * FROM ai_chat_logs WHERE user_id=? AND id=?",
        (user_id, chat_id),
    )
    return df.iloc[0].to_dict() if not df.empty else None


# ======================================================
# PubMed strategy prompt
# ======================================================
def build_pubmed_strategy_prompts(user_text: str):
    system_default = (
        "你是资深医学信息检索专家（PubMed）。"
        "你的任务是把用户给出的研究问题/主题描述，转换为可直接在 PubMed 使用的检索策略。"
        "输出必须结构化，并给出可复制的检索式。"
        "不要编造具体研究结论，只做检索策略。"
    )

    user_prompt = f"""
请根据下面这段文字生成 PubMed 文献检索策略：

【用户描述】
{user_text}

【输出要求】
1) 先用 1-2 句话概括检索目标（PICO/PECO 或核心概念）。
2) 给出“概念拆解表”：每个概念至少包含
   - MeSH 词（如适用，可能多个）
   - 关键词同义词（Title/Abstract 可用）
   - 缩写/拼写变体（如适用）
3) 给出“推荐 PubMed 检索式（可直接复制）”，要求：
   - 使用布尔逻辑 AND/OR
   - 结合 [MeSH Terms]、[Title/Abstract]、必要时 [tiab]
   - 用括号保证逻辑正确
   - 适当使用截词 *（谨慎）
4) 给出“可选限制条件”建议（例如：年份、文献类型、语言、人群、物种），但不要强行加在主检索式里；单独列出。
5) 如用户描述不足以确定关键要素，给出 3-5 个你需要追问的问题（用于改进检索式）。

请用中文输出，检索式部分保持英文标签（PubMed 语法）。
""".strip()

    return system_default, user_prompt


# ======================================================
# UI
# ======================================================
st.set_page_config(layout="wide")
st.title("📚 PubMed 文献检索系统（多用户 + 加密 + 多轮 AI）")

init_db()

# --- Auth UI ---
with st.sidebar:
    st.header("👤 用户")
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None

    if st.session_state["user_id"] is None:
        tab_login, tab_register = st.tabs(["登录", "注册"])

        with tab_login:
            u = st.text_input("用户名", key="login_u")
            p = st.text_input("密码", type="password", key="login_p")
            if st.button("登录"):
                uid = login_user(u, p)
                if uid:
                    st.session_state["user_id"] = uid
                    st.success("登录成功")
                    st.rerun()
                else:
                    st.error("用户名或密码错误")

        with tab_register:
            u2 = st.text_input("新用户名", key="reg_u")
            p2 = st.text_input("新密码（>=6位）", type="password", key="reg_p")
            if st.button("注册"):
                try:
                    uid = create_user(u2, p2)
                    st.session_state["user_id"] = uid
                    st.success("注册并登录成功")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        st.stop()

    else:
        st.write(f"当前用户：**{get_username(st.session_state['user_id'])}**")
        if st.button("退出登录"):
            st.session_state["user_id"] = None
            st.rerun()

user_id = st.session_state["user_id"]

# --- session init ---
if "page" not in st.session_state:
    st.session_state["page"] = 0
if "selected_pmids" not in st.session_state:
    st.session_state["selected_pmids"] = []
if "export_request" not in st.session_state:
    st.session_state["export_request"] = None
if "fav_export_request" not in st.session_state:
    st.session_state["fav_export_request"] = None
if "ai_notice" not in st.session_state:
    st.session_state["ai_notice"] = None

# review multi-turn chat session in UI
if "review_chat_session_id" not in st.session_state:
    st.session_state["review_chat_session_id"] = None

# On start: clear THIS USER's cache only (do not delete favorites)
if "cache_cleared_once" not in st.session_state:
    clear_search_cache(user_id)
    st.session_state["cache_cleared_once"] = True


def add_selected(pmid):
    if pmid not in st.session_state["selected_pmids"]:
        st.session_state["selected_pmids"].append(pmid)


def remove_selected(pmid):
    if pmid in st.session_state["selected_pmids"]:
        st.session_state["selected_pmids"].remove(pmid)


# Navigation (NEW: multi-turn review chat page)
page = st.sidebar.radio(
    "📄 页面",
    ["🔍 文献检索", "📌 我的收藏", "🤖 AI 综述生成（单次）", "🧠 AI 综述对话（多轮）", "💬 AI 对话：PubMed检索策略"],
)

# AI settings (per user)
cfg_saved = load_ai_settings(user_id)
with st.sidebar.expander("🤖 AI 接口设置（加密保存 / 用户隔离）", expanded=False):
    base_url_ui = st.text_input("Base URL", value=cfg_saved["base_url"])
    api_key_ui = st.text_input("API Key（加密保存）", value=cfg_saved["api_key"], type="password")
    model_ui = st.text_input("Model", value=cfg_saved["model"])
    temperature_ui = st.slider("Temperature", 0.0, 1.0, float(cfg_saved["temperature"]), 0.05)
    max_tokens_ui = st.slider("Max tokens", 300, 6000, int(cfg_saved["max_tokens"]), 100)
    system_prompt_ui = st.text_area("System Prompt（可选，叠加到内置）", value=cfg_saved["system_prompt"], height=110)

    if st.button("💾 保存接口设置"):
        try:
            save_ai_settings(user_id, base_url_ui, api_key_ui, model_ui, temperature_ui, max_tokens_ui, system_prompt_ui)
            st.success("已保存（API Key 已加密）")
        except Exception as e:
            st.error(str(e))


# ===============================
# Page: Search
# ===============================
if page == "🔍 文献检索":
    query = st.text_input("关键词", "cancer immunotherapy")
    retmax = st.slider("返回数量", 1, 200, 20)

    col_a, col_b, col_c = st.columns(3)
    year_from = col_a.number_input("起始年份", 1900, 2100, 2015)
    year_to = col_b.number_input("结束年份", 1900, 2100, 2026)
    article_type = col_c.selectbox("文献类型", ["All", "Review", "Clinical Trial", "Meta-Analysis"])

    page_size = st.selectbox("每页显示数量", [5, 10, 20, 50, 100], index=1)

    col_search, col_csv, col_ris = st.columns(3)

    with col_search:
        if st.button("🔍 搜索"):
            results = search_pubmed(query, year_from, year_to, article_type, retmax)
            save_search_results_to_db(user_id, results)
            st.session_state["page"] = 0
            st.session_state["selected_pmids"] = []
            st.session_state["export_request"] = None
            for k in list(st.session_state.keys()):
                if str(k).startswith("sel_"):
                    del st.session_state[k]
            st.rerun()

    with col_csv:
        if st.button("⬇ 导出 CSV"):
            st.session_state["export_request"] = "csv"

    with col_ris:
        if st.button("⬇ 导出 RIS"):
            st.session_state["export_request"] = "ris"

    if st.session_state["export_request"] in ("csv", "ris"):
        if not st.session_state["selected_pmids"]:
            show_dialog("提示", "请先勾选需要导出的参考文献。", "export_request")
        else:
            df_selected = load_articles_by_pmids(st.session_state["selected_pmids"])
            if df_selected.empty:
                st.error("导出失败：本地数据库未找到已选文献记录。请重新搜索后再试。")
            else:
                if st.session_state["export_request"] == "csv":
                    trigger_frontend_download(
                        "selected_articles.csv",
                        "text/csv",
                        df_selected.to_csv(index=False).encode("utf-8-sig"),
                    )
                else:
                    trigger_frontend_download(
                        "selected_articles.ris",
                        "application/x-research-info-systems",
                        generate_ris(df_selected).encode("utf-8"),
                    )
        st.session_state["export_request"] = None

    total_results = get_search_total_count(user_id)
    if total_results > 0:
        total_pages = (total_results - 1) // page_size + 1
        current_page = st.session_state["page"]

        st.write(f"📄 共 {total_results} 篇文献 | 总页数: {total_pages} | 当前页: {current_page + 1}")

        col_prev, col_jump, col_next = st.columns([1, 2, 1])

        if col_prev.button("⬅ 上一页", disabled=(current_page == 0)):
            st.session_state["page"] -= 1
            st.rerun()

        if col_next.button("下一页 ➡", disabled=(current_page >= total_pages - 1)):
            st.session_state["page"] += 1
            st.rerun()

        jump_page = col_jump.number_input("跳转到页码", 1, total_pages, current_page + 1)
        if jump_page - 1 != current_page:
            st.session_state["page"] = jump_page - 1
            st.rerun()

        df_page = load_search_page(user_id, current_page, page_size)
        start_index = current_page * page_size

        for local_i, row in enumerate(df_page.to_dict(orient="records")):
            pmid = row.get("pmid")
            title = row.get("title") or ""
            journal = row.get("journal") or ""
            year = row.get("year") if row.get("year") is not None else ""
            abstract = row.get("abstract") or ""
            doi = row.get("doi") or ""
            pmcid = row.get("pmcid") or ""

            global_no = start_index + local_i + 1
            checked = st.checkbox(
                f"{global_no}. {title}",
                value=(pmid in st.session_state["selected_pmids"]),
                key=f"sel_{pmid}",
            )
            if checked:
                add_selected(pmid)
            else:
                remove_selected(pmid)

            st.markdown(f"**期刊:** {journal} | 年份: {year}")

            if pmid:
                st.markdown(f"🆔 PMID: https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
            if pmcid:
                st.markdown(f"🟢 PMCID: https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/")
            if doi:
                st.markdown(f"🔗 DOI: https://doi.org/{doi}")

            citation = get_citation_count(pmid)
            st.markdown(f"📈 引用次数: {citation if citation is not None else '--'}")

            if abstract:
                with st.expander("📄 查看摘要"):
                    st.markdown(highlight_keywords(abstract, query), unsafe_allow_html=True)

            if st.button("⭐ 收藏", key=f"fav_{pmid}"):
                add_favorite(
                    user_id,
                    {
                        "pmid": pmid,
                        "title": title,
                        "journal": journal,
                        "year": year,
                        "abstract": abstract,
                        "doi": doi,
                        "pmcid": pmcid,
                    },
                )
                st.success("已加入收藏")

            st.markdown("---")
    else:
        st.info("暂无搜索结果，请先搜索。")


# ===============================
# Page: Favorites
# ===============================
elif page == "📌 我的收藏":
    st.subheader("📌 我的收藏")
    fav_df = load_favorites(user_id)

    if not fav_df.empty:
        st.write(f"收藏数量: {len(fav_df)}")
        if st.button("🗑 一键清空收藏"):
            clear_favorites(user_id)
            st.rerun()

        st.markdown("---")
        for _, r in fav_df.iterrows():
            col1, col2 = st.columns([12, 1])
            col1.markdown(f"📄 {r.get('title', '')}")
            if col2.button("❌", key=f"del_{r['pmid']}"):
                remove_favorite(user_id, r["pmid"])
                st.rerun()

        st.markdown("---")
        col_fcsv, col_fris = st.columns(2)
        with col_fcsv:
            if st.button("⬇ 导出收藏 CSV"):
                st.session_state["fav_export_request"] = "csv"
        with col_fris:
            if st.button("⬇ 导出收藏 RIS"):
                st.session_state["fav_export_request"] = "ris"
    else:
        st.write("暂无收藏")
        col_fcsv, col_fris = st.columns(2)
        with col_fcsv:
            if st.button("⬇ 导出收藏 CSV"):
                st.session_state["fav_export_request"] = "csv"
        with col_fris:
            if st.button("⬇ 导出收藏 RIS"):
                st.session_state["fav_export_request"] = "ris"

    if st.session_state["fav_export_request"] in ("csv", "ris"):
        fav_df_now = load_favorites(user_id)
        if fav_df_now.empty:
            show_dialog("提示", "暂无收藏，无法导出。", "fav_export_request")
        else:
            if st.session_state["fav_export_request"] == "csv":
                trigger_frontend_download("favorites.csv", "text/csv", fav_df_now.to_csv(index=False).encode("utf-8-sig"))
            else:
                trigger_frontend_download("favorites.ris", "application/x-research-info-systems", generate_ris(fav_df_now).encode("utf-8"))
        st.session_state["fav_export_request"] = None


# ===============================
# Page: AI review (single-shot, kept)
# ===============================
elif page == "🤖 AI 综述生成（单次）":
    st.subheader("🤖 AI 综述生成（单次输出，保存历史）")

    st.markdown("### 1) 选择输入文献来源")
    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    use_selected = col_s1.checkbox("使用已勾选文献", value=True)
    use_favorites = col_s2.checkbox("使用收藏文献", value=False)

    selected_pmids = st.session_state.get("selected_pmids", [])
    fav_pmids = load_favorite_pmids(user_id)

    pmids = []
    source_tag = []
    if use_selected:
        pmids.extend(selected_pmids)
        source_tag.append("selected")
    if use_favorites:
        pmids.extend(fav_pmids)
        source_tag.append("favorites")

    # unique keep order
    seen = set()
    pmids_unique = []
    for p in pmids:
        if p and p not in seen:
            seen.add(p)
            pmids_unique.append(p)

    col_s3.write(f"当前输入文献数：**{len(pmids_unique)}**")

    max_articles = st.slider("最多输入文献数量（防止上下文过长）", 1, 120, 30)
    pmids_unique = pmids_unique[:max_articles]
    df_input = load_articles_by_pmids(pmids_unique) if pmids_unique else pd.DataFrame()

    with st.expander("📄 预览输入文献（PMID / 标题）"):
        if df_input.empty:
            st.info("暂无输入文献。请先在检索页勾选或收藏文献。")
        else:
            st.dataframe(df_input[["pmid", "title"]], use_container_width=True)

    st.markdown("### 2) 综述参数")
    topic_hint = st.text_input("综述主题提示（可选，不填则自动归纳）", value="")
    user_extra = st.text_area("写作要求（可选）", value="", height=90)

    st.markdown("### 3) 生成综述（保存到本地数据库）")
    if st.button("🚀 开始生成"):
        if df_input.empty:
            st.session_state["ai_notice"] = "没有可用输入文献：请先勾选或收藏文献。"
        else:
            cfg = load_ai_settings(user_id)
            base_url = cfg["base_url"].rstrip("/")
            api_key = cfg["api_key"]
            model = cfg["model"]
            temperature = cfg["temperature"]
            max_tokens = cfg["max_tokens"]
            system_prompt_custom = cfg["system_prompt"]

            if not base_url:
                st.session_state["ai_notice"] = "Base URL 为空，请先在侧边栏保存接口设置。"
            elif not model:
                st.session_state["ai_notice"] = "Model 为空，请先在侧边栏保存接口设置。"
            else:
                sys_prompt = review_system_prompt(system_prompt_custom)
                first_user_prompt = build_review_first_turn(topic_hint, df_input, user_extra=user_extra)

                try:
                    with st.spinner("AI 正在生成综述..."):
                        output = call_chat_completions(
                            base_url=base_url,
                            api_key=api_key,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            system_prompt=sys_prompt,
                            messages=[{"role": "user", "content": first_user_prompt}],
                        )

                    save_ai_review_to_db(
                        user_id=user_id,
                        source=",".join(source_tag) if source_tag else "none",
                        pmids=",".join([str(p) for p in pmids_unique]),
                        topic_hint=topic_hint,
                        base_url=base_url,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_prompt=sys_prompt,
                        user_prompt=first_user_prompt,
                        output=output,
                    )
                    st.success("生成完成，并已保存到本地数据库（ai_reviews）")

                    st.text_area("综述内容（每句句末标注 PMID）", output, height=420)
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        if st.button("⬇ 纯前端下载 TXT"):
                            trigger_frontend_download("ai_review.txt", "text/plain", output.encode("utf-8-sig"))
                    with col_d2:
                        if st.button("⬇ 纯前端下载 MD"):
                            trigger_frontend_download("ai_review.md", "text/markdown", output.encode("utf-8-sig"))
                except Exception as e:
                    st.session_state["ai_notice"] = f"AI 调用失败：{e}"

    if st.session_state.get("ai_notice"):
        show_dialog("提示", st.session_state["ai_notice"], "ai_notice")

    st.markdown("### 4) 历史综述（来自本地数据库）")
    hist = list_ai_reviews(user_id, limit=50)
    if hist.empty:
        st.write("暂无历史记录。")
    else:
        options = hist.apply(lambda r: f"#{r['id']} | {r['created_at']} | {r['source']} | {r['model']} | {r['topic_hint']}", axis=1).tolist()
        sel = st.selectbox("选择一条历史记录", options, index=0)
        sel_id = int(sel.split("|")[0].strip().replace("#", ""))
        item = load_ai_review(user_id, sel_id)
        if item:
            with st.expander("查看详情", expanded=True):
                st.write(f"时间：{item['created_at']}")
                st.write(f"来源：{item['source']}")
                st.write(f"文献 PMID：{(item['pmids'] or '')[:200]}{'...' if item.get('pmids') and len(item['pmids'])>200 else ''}")
                st.write(f"模型：{item['model']} | temp={item['temperature']} | max_tokens={item['max_tokens']}")
                st.text_area("输出", item["output"] or "", height=320)
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    if st.button("⬇ 下载该条 TXT"):
                        trigger_frontend_download(f"ai_review_{sel_id}.txt", "text/plain", (item["output"] or "").encode("utf-8-sig"))
                with col_h2:
                    if st.button("⬇ 下载该条 MD"):
                        trigger_frontend_download(f"ai_review_{sel_id}.md", "text/markdown", (item["output"] or "").encode("utf-8-sig"))


# ===============================
# Page: AI review multi-turn chat (NEW)
# ===============================
elif page == "🧠 AI 综述对话（多轮）":
    st.subheader("🧠 AI 综述对话（多轮）")
    st.caption("基于同一批文献上下文进行连续追问；每句仍需在句末标注 PMID。")

    # Sidebar-ish controls for selecting input
    st.markdown("### 1) 选择对话使用的文献")
    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    use_selected = col_s1.checkbox("使用已勾选文献", value=True, key="chat_use_selected")
    use_favorites = col_s2.checkbox("使用收藏文献", value=False, key="chat_use_fav")

    selected_pmids = st.session_state.get("selected_pmids", [])
    fav_pmids = load_favorite_pmids(user_id)

    pmids = []
    source_tag = []
    if use_selected:
        pmids.extend(selected_pmids)
        source_tag.append("selected")
    if use_favorites:
        pmids.extend(fav_pmids)
        source_tag.append("favorites")

    seen = set()
    pmids_unique = []
    for p in pmids:
        if p and p not in seen:
            seen.add(p)
            pmids_unique.append(p)

    col_s3.write(f"当前输入文献数：**{len(pmids_unique)}**")
    max_articles = st.slider("最多输入文献数量", 1, 120, 30, key="chat_max_articles")
    pmids_unique = pmids_unique[:max_articles]
    df_input = load_articles_by_pmids(pmids_unique) if pmids_unique else pd.DataFrame()

    with st.expander("📄 预览输入文献（PMID / 标题）"):
        if df_input.empty:
            st.info("暂无输入文献。请先在检索页勾选或收藏文献。")
        else:
            st.dataframe(df_input[["pmid", "title"]], use_container_width=True)

    st.markdown("### 2) 开始新对话 / 继续历史对话")
    col_new, col_hist = st.columns([1, 2])

    topic_hint = col_new.text_input("主题提示（可选）", value="", key="chat_topic_hint")
    user_extra = col_new.text_area("首轮写作要求（可选）", value="", height=90, key="chat_user_extra")

    if col_new.button("🆕 用当前文献启动新对话", disabled=df_input.empty):
        if df_input.empty:
            st.session_state["ai_notice"] = "没有可用输入文献：请先勾选或收藏文献。"
        else:
            ctx = build_ai_context(df_input)
            sid = create_review_chat_session(user_id, pmids_unique, topic_hint, ctx)
            st.session_state["review_chat_session_id"] = sid

            # auto send first message to generate initial review
            cfg = load_ai_settings(user_id)
            sys_prompt = review_system_prompt(cfg["system_prompt"])
            first_user_prompt = build_review_first_turn(topic_hint, df_input, user_extra=user_extra)

            try:
                with st.spinner("AI 正在生成首轮综述..."):
                    out = call_chat_completions(
                        base_url=cfg["base_url"].rstrip("/"),
                        api_key=cfg["api_key"],
                        model=cfg["model"],
                        temperature=cfg["temperature"],
                        max_tokens=cfg["max_tokens"],
                        system_prompt=sys_prompt,
                        messages=[{"role": "user", "content": first_user_prompt}],
                    )
                add_review_chat_message(user_id, sid, "user", first_user_prompt)
                add_review_chat_message(user_id, sid, "assistant", out)
                st.success("已创建对话并生成首轮综述。")
                st.rerun()
            except Exception as e:
                st.session_state["ai_notice"] = f"AI 调用失败：{e}"

    # history sessions selector
    hist_df = list_review_chat_sessions(user_id, limit=50)
    if not hist_df.empty:
        options = hist_df.apply(
            lambda r: f"#{r['id']} | {r['updated_at']} | {r['topic_hint'] or '无主题'} | PMIDs:{r['pmids_preview']}",
            axis=1,
        ).tolist()
        sel = col_hist.selectbox("选择历史对话", options, index=0, key="chat_hist_sel")
        sel_id = int(sel.split("|")[0].strip().replace("#", ""))
        if col_hist.button("📌 打开该对话"):
            st.session_state["review_chat_session_id"] = sel_id
            st.rerun()

    st.markdown("---")

    sid = st.session_state.get("review_chat_session_id")
    if sid is None:
        st.info("请先启动新对话或打开一个历史对话。")
    else:
        sess = load_review_chat_session(user_id, sid)
        if not sess:
            st.warning("该对话不存在或无权限。")
        else:
            st.write(f"当前对话：**#{sid}** | 更新时间：{sess.get('updated_at')}")
            with st.expander("查看对话元信息", expanded=False):
                st.write(f"主题提示：{sess.get('topic_hint') or ''}")
                st.write(f"PMIDs：{sess.get('pmids') or ''}")

            messages = load_review_chat_messages(user_id, sid, limit=200)

            # Display chat
            for m in messages:
                if m["role"] == "user":
                    st.chat_message("user").write(m["content"])
                else:
                    st.chat_message("assistant").write(m["content"])

            # Chat input
            user_q = st.chat_input("继续追问（例如：请比较这些研究的局限性；或请提出未来研究方向）")
            if user_q:
                cfg = load_ai_settings(user_id)
                sys_prompt = review_system_prompt(cfg["system_prompt"])

                # Build messages: include "context" as a pinned user message at the start (so model always sees doc context)
                pinned_context = (
                    "以下是本次对话固定使用的文献数据（题目+摘要）。你必须只依据这些信息回答，并继续遵守每句句末 PMID 标注：\n\n"
                    + (sess.get("context") or "")
                )

                model_messages = [{"role": "user", "content": pinned_context}]

                # include previous turns (trim to avoid token blow-up)
                # keep last 10 user+assistant pairs
                trimmed = messages[-20:] if len(messages) > 20 else messages
                for m in trimmed:
                    model_messages.append({"role": m["role"], "content": m["content"]})

                model_messages.append({"role": "user", "content": user_q})

                try:
                    with st.spinner("AI 正在回答..."):
                        out = call_chat_completions(
                            base_url=cfg["base_url"].rstrip("/"),
                            api_key=cfg["api_key"],
                            model=cfg["model"],
                            temperature=cfg["temperature"],
                            max_tokens=cfg["max_tokens"],
                            system_prompt=sys_prompt,
                            messages=model_messages,
                        )
                    add_review_chat_message(user_id, sid, "user", user_q)
                    add_review_chat_message(user_id, sid, "assistant", out)
                    st.rerun()
                except Exception as e:
                    st.session_state["ai_notice"] = f"AI 调用失败：{e}"

            # Export current conversation
            col_e1, col_e2, col_e3 = st.columns([1, 1, 2])
            if col_e1.button("⬇ 导出对话 TXT"):
                txt = ""
                for m in messages:
                    txt += f"[{m['created_at']}] {m['role'].upper()}:\n{m['content']}\n\n"
                trigger_frontend_download(f"review_chat_{sid}.txt", "text/plain", txt.encode("utf-8-sig"))

            if col_e2.button("🧹 清空并关闭当前对话"):
                st.session_state["review_chat_session_id"] = None
                st.rerun()


# ===============================
# Page: PubMed strategy chat (kept)
# ===============================
else:
    st.subheader("💬 AI 对话：生成 PubMed 文献检索策略（保存到本地数据库）")

    st.markdown(
        """
把你的研究问题/主题描述粘贴到下面（越具体越好：人群、干预/暴露、对照、结局、研究类型等）。  
点击生成后，AI 会输出：概念拆解 + 可复制的 PubMed 检索式 + 可选限制条件 + 追问问题。
"""
    )

    user_text = st.text_area("你的描述", height=160, placeholder="例如：我想检索PD-1/PD-L1抑制剂在非小细胞肺癌一线治疗中的疗效与安全性...")

    col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
    with col_c1:
        if st.button("🧠 生成检索策略"):
            if not user_text.strip():
                st.session_state["ai_notice"] = "请输入一段描述后再生成。"
            else:
                cfg = load_ai_settings(user_id)
                base_url = cfg["base_url"].rstrip("/")
                api_key = cfg["api_key"]
                model = cfg["model"]
                temperature = cfg["temperature"]
                max_tokens = cfg["max_tokens"]
                system_prompt_custom = cfg["system_prompt"]

                if not base_url:
                    st.session_state["ai_notice"] = "Base URL 为空，请先在侧边栏保存接口设置。"
                elif not model:
                    st.session_state["ai_notice"] = "Model 为空，请先在侧边栏保存接口设置。"
                else:
                    sys_default, user_prompt = build_pubmed_strategy_prompts(user_text)
                    system_prompt = sys_default + ("\n\n" + system_prompt_custom.strip() if system_prompt_custom.strip() else "")

                    try:
                        with st.spinner("AI 正在生成 PubMed 检索策略..."):
                            out = call_chat_completions(
                                base_url=base_url,
                                api_key=api_key,
                                model=model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                system_prompt=system_prompt,
                                messages=[{"role": "user", "content": user_prompt}],
                            )

                        save_chat_log(
                            user_id=user_id,
                            chat_type="pubmed_strategy",
                            base_url=base_url,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            system_prompt=system_prompt,
                            user_input=user_text,
                            assistant_output=out,
                        )

                        st.session_state["chat_last_output"] = out
                        st.success("已生成并保存到本地数据库（ai_chat_logs）")
                    except Exception as e:
                        st.session_state["ai_notice"] = f"AI 调用失败：{e}"

    with col_c2:
        if st.button("🧹 清空当前输出"):
            st.session_state["chat_last_output"] = ""

    with col_c3:
        st.caption("提示：可在侧边栏配置 Base URL / Key / Model 等（用户隔离 + Key 加密保存）。")

    if st.session_state.get("ai_notice"):
        show_dialog("提示", st.session_state["ai_notice"], "ai_notice")

    st.markdown("### 当前输出")
    if st.session_state.get("chat_last_output", "").strip():
        st.text_area("PubMed 检索策略", st.session_state["chat_last_output"], height=420)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if st.button("⬇ 纯前端下载 TXT"):
                trigger_frontend_download("pubmed_search_strategy.txt", "text/plain", st.session_state["chat_last_output"].encode("utf-8-sig"))
        with col_d2:
            if st.button("⬇ 纯前端下载 MD"):
                trigger_frontend_download("pubmed_search_strategy.md", "text/markdown", st.session_state["chat_last_output"].encode("utf-8-sig"))
    else:
        st.info("暂无输出。输入描述后点击“生成检索策略”。")

    st.markdown("### 历史对话（来自本地数据库）")
    hist = list_chat_logs(user_id, "pubmed_strategy", limit=50)
    if hist.empty:
        st.write("暂无历史记录。")
    else:
        options = hist.apply(lambda r: f"#{r['id']} | {r['created_at']} | {r['model']} | {str(r['user_input'])[:40]}", axis=1).tolist()
        sel = st.selectbox("选择一条历史记录", options, index=0)
        sel_id = int(sel.split("|")[0].strip().replace("#", ""))
        item = load_chat_log(user_id, sel_id)
        if item:
            with st.expander("查看历史详情", expanded=True):
                st.write(f"时间：{item['created_at']}")
                st.write(f"模型：{item['model']} | temp={item['temperature']} | max_tokens={item['max_tokens']}")
                st.text_area("用户输入", item["user_input"] or "", height=140)
                st.text_area("AI 输出", item["assistant_output"] or "", height=320)

                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    if st.button("⬇ 下载该条 TXT"):
                        trigger_frontend_download(f"pubmed_strategy_{sel_id}.txt", "text/plain", (item["assistant_output"] or "").encode("utf-8-sig"))
                with col_h2:
                    if st.button("⬇ 下载该条 MD"):
                        trigger_frontend_download(f"pubmed_strategy_{sel_id}.md", "text/markdown", (item["assistant_output"] or "").encode("utf-8-sig"))
