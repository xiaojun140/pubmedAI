import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import sqlite3
import base64
import time
from datetime import datetime

DB_FILE = "articles.db"


# ===============================
# 初始化数据库
# ===============================
def init_db():
    conn = sqlite3.connect(DB_FILE)
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

    # AI 设置（单行保存）
    c.execute("""
        CREATE TABLE IF NOT EXISTS ai_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            base_url TEXT,
            api_key TEXT,
            model TEXT,
            temperature REAL,
            max_tokens INTEGER,
            system_prompt TEXT
        )
    """)

    # AI 综述历史
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

    # AI 对话历史（用于“生成 PubMed 检索策略”）
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

    # Ensure ai_settings schema is compatible with this app
    try:
        migrate_ai_settings_schema(conn)
    except Exception:
        pass

    conn.commit()
    conn.close()


# ===============================
# AI 设置：DB 读写
# ===============================

def _ai_settings_cols(conn) -> list[str]:
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(ai_settings)")
        return [r[1] for r in cur.fetchall()]
    except Exception:
        return []


def migrate_ai_settings_schema(conn):
    """Make ai_settings compatible with this app (single-row, id=1)."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_settings'")
    if not cur.fetchone():
        return

    cols = _ai_settings_cols(conn)
    # Desired legacy-compatible schema: id=1 + plaintext api_key
    if "id" in cols and "api_key" in cols:
        return

    legacy = f"ai_settings_legacy_{int(time.time())}"
    # Rename old table
    cur.execute(f"ALTER TABLE ai_settings RENAME TO {legacy}")

    # Recreate table with expected schema
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            base_url TEXT,
            api_key TEXT,
            model TEXT,
            temperature REAL,
            max_tokens INTEGER,
            system_prompt TEXT
        )
    """)

    # Best-effort copy (API key may be unavailable if old schema used api_key_enc)
    base_expr = "base_url" if "base_url" in cols else "NULL"
    api_expr  = "api_key" if "api_key" in cols else "''"
    model_expr = "model" if "model" in cols else "NULL"
    temp_expr = "temperature" if "temperature" in cols else "0.3"
    max_expr = "max_tokens" if "max_tokens" in cols else "1500"
    sys_expr = "system_prompt" if "system_prompt" in cols else "''"

    try:
        cur.execute(
            f"""INSERT INTO ai_settings (id, base_url, api_key, model, temperature, max_tokens, system_prompt)
                SELECT 1, {base_expr}, {api_expr}, {model_expr}, {temp_expr}, {max_expr}, {sys_expr}
                FROM {legacy}
                LIMIT 1
            """
        )
    except Exception:
        # If copy fails, keep defaults; user can re-enter settings
        pass


def load_ai_settings():
    conn = sqlite3.connect(DB_FILE)
    try:
        # Ensure schema is compatible (handles old DB files)
        try:
            migrate_ai_settings_schema(conn)
        except Exception:
            pass

        c = conn.cursor()
        cols = _ai_settings_cols(conn)

        if "id" in cols:
            # single-row mode
            if "api_key" in cols:
                c.execute(
                    "SELECT base_url, api_key, model, temperature, max_tokens, system_prompt "
                    "FROM ai_settings WHERE id=1"
                )
            else:
                # legacy encrypted column cannot be read without its decryptor; force re-enter
                c.execute(
                    "SELECT base_url, '' as api_key, model, temperature, max_tokens, system_prompt "
                    "FROM ai_settings WHERE id=1"
                )
            row = c.fetchone()
        else:
            # Unknown schema: return defaults
            row = None

        if not row:
            return {
                "base_url": "https://api.openai.com/v1",
                "api_key": "",
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 1500,
                "system_prompt": ""
            }

        return {
            "base_url": row[0] or "https://api.openai.com/v1",
            "api_key": row[1] or "",
            "model": row[2] or "gpt-4o-mini",
            "temperature": float(row[3]) if row[3] is not None else 0.3,
            "max_tokens": int(row[4]) if row[4] is not None else 1500,
            "system_prompt": row[5] or ""
        }
    finally:
        conn.close()


def save_ai_settings(base_url, api_key, model, temperature, max_tokens, system_prompt):
    conn = sqlite3.connect(DB_FILE)
    try:
        try:
            migrate_ai_settings_schema(conn)
        except Exception:
            pass

        c = conn.cursor()
        cols = _ai_settings_cols(conn)

        # API Key 不回显：如果留空则保留旧 Key
        if not (api_key or "").strip():
            try:
                old = load_ai_settings().get("api_key", "")
            except Exception:
                old = ""
            api_key = old

        if "id" not in cols:
            # If schema is still unexpected, re-migrate and re-check
            migrate_ai_settings_schema(conn)
            cols = _ai_settings_cols(conn)

        c.execute("""
            INSERT INTO ai_settings (id, base_url, api_key, model, temperature, max_tokens, system_prompt)
            VALUES (1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                base_url=excluded.base_url,
                api_key=excluded.api_key,
                model=excluded.model,
                temperature=excluded.temperature,
                max_tokens=excluded.max_tokens,
                system_prompt=excluded.system_prompt
        """, (
            (base_url or "").strip().rstrip("/"),
            (api_key or "").strip(),
            (model or "").strip(),
            float(temperature),
            int(max_tokens),
            system_prompt or ""
        ))

        conn.commit()
    finally:
        conn.close()
# ===============================
# AI 综述：DB 读写
# ===============================
def save_ai_review_to_db(source, pmids, topic_hint, base_url, model, temperature, max_tokens, system_prompt, user_prompt, output):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO ai_reviews (
            created_at, source, pmids, topic_hint,
            base_url, model, temperature, max_tokens,
            system_prompt, user_prompt, output
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
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
        output
    ))

    new_id = c.lastrowid  # 取回自增 id（用于“改写链路”追踪）

    # Ensure ai_settings schema is compatible with this app
    try:
        migrate_ai_settings_schema(conn)
    except Exception:
        pass

    conn.commit()
    conn.close()
    return new_id


def list_ai_reviews(limit=50):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT id, created_at, source, topic_hint, model
        FROM ai_reviews
        ORDER BY id DESC
        LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return df


def load_ai_review(review_id: int):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT *
        FROM ai_reviews
        WHERE id = ?
    """, conn, params=(review_id,))
    conn.close()
    if df.empty:
        return None
    return df.iloc[0].to_dict()


# ===============================
# AI 对话：DB 读写
# ===============================
def save_chat_log(chat_type, base_url, model, temperature, max_tokens, system_prompt, user_input, assistant_output):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO ai_chat_logs (
            created_at, chat_type, base_url, model, temperature, max_tokens,
            system_prompt, user_input, assistant_output
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        chat_type,
        base_url,
        model,
        float(temperature),
        int(max_tokens),
        system_prompt,
        user_input,
        assistant_output
    ))
    # Ensure ai_settings schema is compatible with this app
    try:
        migrate_ai_settings_schema(conn)
    except Exception:
        pass

    conn.commit()
    conn.close()


def list_chat_logs(chat_type, limit=50):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT id, created_at, model, user_input
        FROM ai_chat_logs
        WHERE chat_type = ?
        ORDER BY id DESC
        LIMIT ?
    """, conn, params=(chat_type, limit))
    conn.close()
    return df


def load_chat_log(chat_id: int):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT *
        FROM ai_chat_logs
        WHERE id = ?
    """, conn, params=(chat_id,))
    conn.close()
    if df.empty:
        return None
    return df.iloc[0].to_dict()


# ===============================
# 搜索缓存：启动清空（不删除收藏）
# ===============================
def clear_search_cache():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        DELETE FROM articles
        WHERE pmid IN (SELECT pmid FROM search_cache)
          AND pmid NOT IN (SELECT pmid FROM favorites)
    """)

    c.execute("DELETE FROM search_cache")
    # Ensure ai_settings schema is compatible with this app
    try:
        migrate_ai_settings_schema(conn)
    except Exception:
        pass

    conn.commit()
    conn.close()


def save_search_results_to_db(articles):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        DELETE FROM articles
        WHERE pmid IN (SELECT pmid FROM search_cache)
          AND pmid NOT IN (SELECT pmid FROM favorites)
    """)
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
            INSERT INTO articles (pmid, title, journal, year, abstract, doi, pmcid)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pmid) DO UPDATE SET
                title=excluded.title,
                journal=excluded.journal,
                year=excluded.year,
                abstract=excluded.abstract,
                doi=excluded.doi,
                pmcid=excluded.pmcid
        """, (
            pmid,
            article.get("title"),
            article.get("journal"),
            year_val,
            article.get("abstract"),
            article.get("doi"),
            article.get("pmcid")
        ))

        c.execute("""
            INSERT OR REPLACE INTO search_cache (idx, pmid)
            VALUES (?, ?)
        """, (i, pmid))

    # Ensure ai_settings schema is compatible with this app
    try:
        migrate_ai_settings_schema(conn)
    except Exception:
        pass

    conn.commit()
    conn.close()


def get_search_total_count():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM search_cache")
    total = c.fetchone()[0]
    conn.close()
    return total


def load_search_page(page, page_size):
    offset = page * page_size
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT a.*
        FROM search_cache s
        JOIN articles a ON a.pmid = s.pmid
        ORDER BY s.idx
        LIMIT ? OFFSET ?
    """, conn, params=(page_size, offset))
    conn.close()
    return df


def load_articles_by_pmids(pmids):
    if not pmids:
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(pmids))
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        f"SELECT * FROM articles WHERE pmid IN ({placeholders})",
        conn,
        params=list(pmids)
    )
    conn.close()

    order_map = {pmid: i for i, pmid in enumerate(pmids)}
    if not df.empty:
        df["__ord"] = df["pmid"].map(order_map)
        df = df.sort_values("__ord").drop(columns="__ord")
    return df


# ===============================
# 收藏数据库操作
# ===============================
def add_favorite(article):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    year_val = None
    y = article.get("year")
    if y and str(y).isdigit():
        year_val = int(y)

    c.execute("""
        INSERT INTO articles (pmid, title, journal, year, abstract, doi, pmcid)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pmid) DO UPDATE SET
            title=excluded.title,
            journal=excluded.journal,
            year=excluded.year,
            abstract=excluded.abstract,
            doi=excluded.doi,
            pmcid=excluded.pmcid
    """, (
        article.get("pmid"),
        article.get("title"),
        article.get("journal"),
        year_val,
        article.get("abstract"),
        article.get("doi"),
        article.get("pmcid")
    ))

    c.execute("INSERT OR IGNORE INTO favorites VALUES (?)", (article.get("pmid"),))
    # Ensure ai_settings schema is compatible with this app
    try:
        migrate_ai_settings_schema(conn)
    except Exception:
        pass

    conn.commit()
    conn.close()


def remove_favorite(pmid):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM favorites WHERE pmid = ?", (pmid,))
    # Ensure ai_settings schema is compatible with this app
    try:
        migrate_ai_settings_schema(conn)
    except Exception:
        pass

    conn.commit()
    conn.close()


def clear_favorites():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM favorites")
    # Ensure ai_settings schema is compatible with this app
    try:
        migrate_ai_settings_schema(conn)
    except Exception:
        pass

    conn.commit()
    conn.close()


def load_favorites():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT a.*
        FROM articles a
        JOIN favorites f ON a.pmid = f.pmid
    """, conn)
    conn.close()
    return df


def load_favorite_pmids():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT pmid FROM favorites", conn)
    conn.close()
    return df["pmid"].tolist()


# ===============================
# RIS 导出
# ===============================
def generate_ris(df):
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


# ===============================
# 获取引用次数
# ===============================
@st.cache_data(show_spinner=False)
def get_citation_count(pmid):
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/PMID:{pmid}?fields=citationCount"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("citationCount", 0)
    except:
        pass
    return None


# ===============================
# 摘要关键词高亮
# ===============================
def highlight_keywords(text, keywords):
    if not text:
        return ""
    highlighted = text
    for word in keywords.split():
        w = word.strip()
        if w:
            highlighted = highlighted.replace(w, f"<span style='background-color:yellow'>{w}</span>")
    return highlighted


# ===============================
# PubMed 搜索
# ===============================
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
        timeout=15
    ).json()

    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    fetch = requests.get(
        base + "efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        timeout=30
    )

    root = ET.fromstring(fetch.content)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        articles.append({
            "pmid": article.findtext(".//PMID"),
            "title": article.findtext(".//ArticleTitle"),
            "journal": article.findtext(".//Journal/Title"),
            "year": article.findtext(".//PubDate/Year"),
            "abstract": " ".join([a.text for a in article.findall(".//AbstractText") if a.text]),
            "doi": next((a.text for a in article.findall(".//ArticleId") if a.attrib.get("IdType") == "doi"), None),
            "pmcid": next((a.text for a in article.findall(".//ArticleId") if a.attrib.get("IdType") == "pmc"), None)
        })
    return articles


# ===============================
# 纯前端触发下载（兼容老版本 Streamlit）
# ===========================
