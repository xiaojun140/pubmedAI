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

    new_id = c.lastrowid  # ✅ 取回自增 id（用于追踪改写来源）

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


def clear_ai_reviews():
    """Delete all AI review history records."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM ai_reviews")
    conn.commit()
    conn.close()


def clear_chat_logs(chat_type: str | None = None):
    """Delete AI chat logs. If chat_type provided, only delete that type."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if chat_type:
        c.execute("DELETE FROM ai_chat_logs WHERE chat_type = ?", (chat_type,))
    else:
        c.execute("DELETE FROM ai_chat_logs")
    conn.commit()
    conn.close()


def delete_ai_review(review_id: int):
    """Delete a single AI review record by id."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM ai_reviews WHERE id = ?", (int(review_id),))
    conn.commit()
    conn.close()


def delete_chat_log(chat_id: int):
    """Delete a single AI chat log record by id."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM ai_chat_logs WHERE id = ?", (int(chat_id),))
    conn.commit()
    conn.close()


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
# ===============================
def trigger_frontend_download(filename: str, mime: str, data_bytes: bytes):
    if "dl_nonce" not in st.session_state:
        st.session_state["dl_nonce"] = 0
    st.session_state["dl_nonce"] += 1

    nonce = f"{st.session_state['dl_nonce']}_{int(time.time()*1000)}"
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


# ===============================
# Dialog helper
# ===============================
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


# ===============================
# AI：调用（OpenAI-compatible Chat Completions）
# ===============================
def call_chat_completions(base_url, api_key, model, temperature, max_tokens, system_prompt, user_prompt):
    base_url = (base_url or "").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url
    else:
        base_url = base_url + "/v1"
    url = f"{base_url}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt}
        ]
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"接口返回 {r.status_code}: {r.text[:500]}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"解析返回失败：{data}")



# ===============================
# API：获取模型列表（OpenAI-compatible）
# ===============================
def fetch_available_models(base_url: str, api_key: str) -> list[str]:
    """
    Try GET {base_url}/models (OpenAI-compatible).
    If 404 and base_url doesn't end with /v1, retry with {base_url}/v1/models.
    Returns a list of model IDs.
    """
    base_url = (base_url or "").rstrip("/")
    api_key = (api_key or "").strip()
    if not base_url:
        raise RuntimeError("Base URL 为空")
    if not api_key:
        raise RuntimeError("API Key 为空")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _do(url: str):
        r = requests.get(url, headers=headers, timeout=30)
        return r

    url1 = f"{base_url}/models"
    r = _do(url1)

    # retry with /v1 if likely missing
    if r.status_code == 404 and not base_url.endswith("/v1"):
        url2 = f"{base_url}/v1/models"
        r2 = _do(url2)
        if r2.status_code == 200:
            r = r2

    if r.status_code != 200:
        raise RuntimeError(f"获取模型失败 {r.status_code}: {r.text[:300]}")
    data = r.json()
    items = data.get("data", []) if isinstance(data, dict) else []
    models = []
    for it in items:
        mid = (it or {}).get("id")
        if mid:
            models.append(str(mid))
    models = sorted(set(models), key=lambda x: (0 if x.startswith("gpt") else 1, x))
    if not models:
        raise RuntimeError("未获取到任何模型（接口返回 data 为空）")
    return models

# ===============================
# AI：综述 prompt 构建
# ===============================

# ======================================================
# API: list available models (OpenAI-compatible /models)
# ======================================================
def fetch_available_models(base_url: str, api_key: str, timeout: int = 15) -> list[str]:
    """
    Try to fetch model list from an OpenAI-compatible endpoint.
    Supports both:
      - base_url already includes /v1  -> GET {base_url}/models
      - base_url is a root            -> GET {base_url}/v1/models
    Returns a list of model ids.
    """
    base_url = (base_url or "").strip().rstrip("/")
    api_key = (api_key or "").strip()

    if not base_url:
        raise RuntimeError("Base URL 不能为空")
    if not api_key:
        raise RuntimeError("API Key 不能为空")

    candidates = [f"{base_url}/models"]
    if not base_url.endswith("/v1"):
        candidates.append(f"{base_url}/v1/models")

    headers = {"Authorization": f"Bearer {api_key}"}

    last_err = None
    for url in candidates:
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code != 200:
                last_err = RuntimeError(f"获取模型失败 {r.status_code}: {r.text[:200]}")
                continue
            data = r.json()

            models = []
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                for item in data["data"]:
                    if isinstance(item, dict):
                        mid = item.get("id") or item.get("name")
                        if mid:
                            models.append(str(mid))
            # Some gateways might return plain list
            if not models and isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        models.append(item)
                    elif isinstance(item, dict) and (item.get("id") or item.get("name")):
                        models.append(str(item.get("id") or item.get("name")))

            models = sorted(list(dict.fromkeys(models)))
            if not models:
                raise RuntimeError("接口 /models 返回为空或不兼容，请检查网关是否支持 OpenAI /models 格式")
            return models
        except Exception as e:
            last_err = e

    raise RuntimeError(str(last_err) if last_err else "无法获取模型列表")


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


def build_review_prompts(topic_hint: str, df: pd.DataFrame, user_extra: str = ""):
    context = build_ai_context(df)

    # ===== System Prompt =====
    system_default = (
        "你是一名严谨的医学/生命科学综述写作助手。"
        "你必须只依据用户提供的参考文献信息（题目与摘要）进行归纳，避免臆测。"
        "输出为中文。"
        "非常重要：每一句话末尾必须用括号标注 PMID 作为来源，格式严格为 (PMID:12345678) 或 (PMID:123; PMID:456)。"
        "如果一句话综合多篇文献，列出多个 PMID。"
        "不要在句中插入 PMID，只能句末标注。"
        "不要输出参考文献列表（用户将用 EndNote 处理）。"
    )

    # ===== 处理主题 =====
    topic_value = topic_hint.strip() or "由文献内容自动归纳主题"

    # ===== 默认写作要求 =====
    default_requirement = (
        "1 建议结构：背景与问题  关键机制或证据  临床、应用或研究进展 局限性 未来方向\n"
        "2 尽量做到归纳对比，避免逐篇复述。\n"
        "3 任何一句话都必须在句末标注 PMID。\n"
        "4 禁止编造，若文献中无信息支撑，请使用谨慎表达，并仍标注来源 PMID。\n"
        "5 篇幅：约 800~1500 字，可根据文献数量适当调整。"
    )

    # ===== 如果用户填写了自定义要求，则覆盖默认 =====
    extra_requirement = user_extra.strip() or default_requirement

    # ===== User Prompt =====
    user_prompt = f"""
请基于下面提供的文献题目与摘要，围绕主题生成一篇结构清晰的综述。

【主题/方向】
{topic_value}

【写作要求】
{extra_requirement}

【文献数据】
{context}
""".strip()

    return system_default, user_prompt

def build_review_revision_prompts(
    instruction: str,
    original_review: str,
    df: pd.DataFrame,
    topic_hint: str = ""
):
    context = build_ai_context(df)

    system_default = (
        "你是一名严谨的医学/生命科学综述写作助手。"
        "你将收到：1) 原综述 2) 修改指令 3) 文献题目与摘要。"
        "你必须只依据文献题目与摘要修改原综述，避免臆测。"
        "输出为中文。"
        "非常重要：每一句话末尾必须用括号标注 PMID 作为来源，格式严格为 "
        "(PMID:12345678 IF: NA NA NA) 或 (PMID:123; PMID:456)。"
        "不要在句中插入 PMID，只能句末标注。"
        "不得编造新的 PMID；如需新增表述，必须能从给定文献中找到依据并标注相应 PMID。"
        "不要输出参考文献列表。"
    )

    topic_value = topic_hint.strip() or "沿用原综述主题"

    user_prompt = f"""
请按“修改指令”对“原综述”进行改写，输出一份“修改后的完整综述”（不是修改建议、不是对比清单）。

【主题/方向】
{topic_value}

【修改指令】
{instruction.strip()}

【原综述】
{original_review.strip()}

【文献数据（仅允许依据这些内容）】
{context}
""".strip()

    return system_default, user_prompt





# ===============================
# AI：PubMed 检索策略 prompt 构建
# ===============================
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


# ===============================
# UI
# ===============================
st.set_page_config(layout="wide")
st.title("📚 PubMed 文献检索系统")

init_db()

# 启动时清上次搜索缓存（不删收藏）
if "cache_cleared_once" not in st.session_state:
    clear_search_cache()
    st.session_state["cache_cleared_once"] = True

# session 初始化
if "page" not in st.session_state:
    st.session_state["page"] = 0
if "selected_pmids" not in st.session_state:
    st.session_state["selected_pmids"] = []
if "export_request" not in st.session_state:
    st.session_state["export_request"] = None
if "fav_export_request" not in st.session_state:
    st.session_state["fav_export_request"] = None
if "ai_last_output" not in st.session_state:
    st.session_state["ai_last_output"] = ""
if "ai_last_review_id" not in st.session_state:
    st.session_state["ai_last_review_id"] = None
if "ai_last_pmids" not in st.session_state:
    st.session_state["ai_last_pmids"] = []
if "ai_last_topic_hint" not in st.session_state:
    st.session_state["ai_last_topic_hint"] = ""
if "ai_notice" not in st.session_state:
    st.session_state["ai_notice"] = None
if "chat_last_output" not in st.session_state:
    st.session_state["chat_last_output"] = ""


# 页面导航（新增：💬 AI 对话）
page = st.sidebar.radio("📄 页面", ["💬 检索策略生成", "🔍 文献检索", "📌 我的收藏", "🤖 AI 综述生成"])


def add_selected(pmid):
    if pmid not in st.session_state["selected_pmids"]:
        st.session_state["selected_pmids"].append(pmid)


def remove_selected(pmid):
    if pmid in st.session_state["selected_pmids"]:
        st.session_state["selected_pmids"].remove(pmid)


# ===============================
# 侧边栏：AI 设置（所有 AI 页面复用）
# ===============================
cfg_saved = load_ai_settings()
with st.sidebar.expander("🤖 AI 接口设置（保存到本地数据库）", expanded=False):
    # IMPORTANT: add keys so other pages/dialogs can programmatically update (e.g., model picker)
    base_url_ui = st.text_input("Base URL", value=cfg_saved["base_url"], key="api_base_url")
    api_key_ui = st.text_input(
        "API Key（不支持回显；留空=不修改）",
        value="",
        type="password",
        key="api_key_input",
        placeholder="已保存（不可回显），如需更换请在此粘贴新 Key",
    )
    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        model_ui = st.text_input("Model", value=cfg_saved["model"], key="api_model")
    with col_m2:
        if st.button("📡 获取模型", key="sidebar_fetch_models"):
            try:
                api_key_for_fetch = (st.session_state.get("api_key_input") or "").strip() or (cfg_saved.get("api_key") or "").strip()
                models = fetch_available_models((st.session_state.get("api_base_url","").strip() or base_url_ui.strip()), api_key_for_fetch)
                st.session_state["model_list_cache"] = models
                # default select current
                cur = st.session_state.get("api_model") or (models[0] if models else "")
                st.session_state["model_selected_tmp"] = cur if cur in models else (models[0] if models else "")
                st.session_state["open_model_dialog"] = True
            except Exception as e:
                st.error(str(e))

    # model picker dialog (shared with API settings page)
    if st.session_state.get("open_model_dialog"):
        @st.dialog("选择模型")
        def _pick_model_dialog_sidebar():
            models = st.session_state.get("model_list_cache") or []
            if not models:
                st.error("模型列表为空，请重新点击“获取模型”。")
                if st.button("关闭", key="dlg_close_empty_sidebar"):
                    st.session_state["open_model_dialog"] = False
                    st.rerun()
                return

            # default index
            cur = st.session_state.get("api_model") or models[0]
            try:
                idx = models.index(cur)
            except Exception:
                idx = 0

            choice = st.selectbox("API 支持的模型", models, index=idx, key="model_selected_tmp")
            c1, c2 = st.columns(2)
            if c1.button("✅ 使用该模型", key="dlg_use_model_sidebar"):
                st.session_state["api_model"] = choice
                st.session_state["api_model_page"] = choice
                st.session_state["open_model_dialog"] = False
                st.rerun()
            if c2.button("取消", key="dlg_cancel_sidebar"):
                st.session_state["open_model_dialog"] = False
                st.rerun()

        _pick_model_dialog_sidebar()
    temperature_ui = st.slider("Temperature", 0.0, 1.0, float(cfg_saved["temperature"]), 0.05, key="api_temp")
    max_tokens_ui = st.slider("Max tokens", 300, 6000, int(cfg_saved["max_tokens"]), 100, key="api_max_tokens")
    system_prompt_ui = st.text_area("System Prompt（可选，留空使用内置）", value=cfg_saved["system_prompt"], height=110, key="api_sys_prompt")

    if st.button("💾 保存接口设置", key="api_save_btn_sidebar"):
        save_ai_settings(base_url_ui, api_key_ui, model_ui, temperature_ui, max_tokens_ui, system_prompt_ui)
        st.success("已保存到本地数据库（ai_settings）")


if page == "🔍 文献检索":

    query = st.text_input("关键词", "cancer immunotherapy")
    retmax = st.slider("返回数量", 1, 200, 20)

    col_a, col_b, col_c = st.columns(3)
    year_from = col_a.number_input("起始年份", 1900, 2100, 2015)
    year_to = col_b.number_input("结束年份", 1900, 2100, 2026)
    article_type = col_c.selectbox("文献类型", ["All", "Review", "Clinical Trial", "Meta-Analysis"])

    page_size = st.selectbox("每页显示数量", [5, 10, 20,50,100], index=1)

    col_search, col_csv, col_ris = st.columns(3)

    with col_search:
        if st.button("🔍 搜索"):
            results = search_pubmed(query, year_from, year_to, article_type, retmax)
            save_search_results_to_db(results)
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

    # 搜索导出：未选中 -> 弹窗；已选中 -> 纯前端下载
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
                        "selected_articles.csv", "text/csv",
                        df_selected.to_csv(index=False).encode("utf-8-sig")
                    )
                else:
                    trigger_frontend_download(
                        "selected_articles.ris", "application/x-research-info-systems",
                        generate_ris(df_selected).encode("utf-8")
                    )
        st.session_state["export_request"] = None

    # 分页展示（从 DB 读）
    total_results = get_search_total_count()
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

        df_page = load_search_page(current_page, page_size)
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
                key=f"sel_{pmid}"
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
                add_favorite({
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "year": year,
                    "abstract": abstract,
                    "doi": doi,
                    "pmcid": pmcid
                })
                st.success("已加入收藏")

            st.markdown("---")
    else:
        st.info("暂无搜索结果，请先搜索。")


# ===============================
# 页面：我的收藏（含纯前端导出）
# ===============================
elif page == "📌 我的收藏":
    st.subheader("📌 我的收藏")
    fav_df = load_favorites()

    if not fav_df.empty:
        st.write(f"收藏数量: {len(fav_df)}")
        if st.button("🗑 一键清空收藏"):
            clear_favorites()
            st.rerun()

        st.markdown("---")
        for _, r in fav_df.iterrows():
            col1, col2 = st.columns([12, 1])
            col1.markdown(f"📄 {r.get('title', '')}")
            if col2.button("❌", key=f"del_{r['pmid']}"):
                remove_favorite(r["pmid"])
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
        fav_df_now = load_favorites()
        if fav_df_now.empty:
            show_dialog("提示", "暂无收藏，无法导出。", "fav_export_request")
        else:
            if st.session_state["fav_export_request"] == "csv":
                trigger_frontend_download("favorites.csv", "text/csv", fav_df_now.to_csv(index=False).encode("utf-8-sig"))
            else:
                trigger_frontend_download("favorites.ris", "application/x-research-info-systems", generate_ris(fav_df_now).encode("utf-8"))
        st.session_state["fav_export_request"] = None


# ===============================
# 页面：AI 综述生成（保存到DB）
# ===============================
elif page == "🤖 AI 综述生成":
    st.subheader("🤖 AI 综述生成（每句标注 PMID，结果保存到本地数据库）")

    st.markdown("### 1) 选择输入文献来源")
    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    use_selected = col_s1.checkbox("使用已勾选文献", value=True)
    use_favorites = col_s2.checkbox("使用收藏文献", value=False)

    selected_pmids = st.session_state.get("selected_pmids", [])
    fav_pmids = load_favorite_pmids()

    pmids = []
    source_tag = []
    if use_selected:
        pmids.extend(selected_pmids)
        source_tag.append("selected")
    if use_favorites:
        pmids.extend(fav_pmids)
        source_tag.append("favorites")

    # 去重保序
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
    user_extra = st.text_area("写作要求（可选）", value="可选，不填则自动按内置指令生成", height=90)

    st.markdown("### 3) 生成综述（自动保存到本地数据库）")
    if st.button("🚀 开始生成"):
        if df_input.empty:
            st.session_state["ai_notice"] = "没有可用输入文献：请先勾选或收藏文献。"
        else:
            cfg = load_ai_settings()
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
                sys_default, user_prompt = build_review_prompts(topic_hint, df_input, user_extra=user_extra)
                system_prompt = sys_default + ("\n\n" + system_prompt_custom.strip() if system_prompt_custom.strip() else "")

                try:
                    with st.spinner("AI 正在生成综述..."):
                        output = call_chat_completions(
                            base_url=base_url,
                            api_key=api_key,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt
                        )
                    st.session_state["ai_last_output"] = output

                    new_id = save_ai_review_to_db(
                        source=",".join(source_tag) if source_tag else "none",
                        pmids=",".join([str(p) for p in pmids_unique]),
                        topic_hint=topic_hint,
                        base_url=base_url,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        output=output
                    )
                    st.session_state["ai_last_review_id"] = new_id  # ✅ 记录本次生成的 review_id
                    # ✅ 记录本次生成使用的 PMID/主题，并保持“检索页已勾选文献”不丢失（便于重复生成）
                    st.session_state["ai_last_pmids"] = list(pmids_unique)
                    st.session_state["ai_last_topic_hint"] = topic_hint
                    st.session_state["selected_pmids"] = list(pmids_unique)
                    st.success("生成完成，并已保存到本地数据库（ai_reviews）")
                except Exception as e:
                    st.session_state["ai_notice"] = f"AI 调用失败：{e}"

    if st.session_state.get("ai_notice"):
        show_dialog("提示", st.session_state["ai_notice"], "ai_notice")

    st.markdown("### 4) 最新生成结果")
    if st.session_state.get("ai_last_output", "").strip():
        st.text_area("综述内容（每句句末标注 PMID）", st.session_state["ai_last_output"], height=420)
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if st.button("⬇ 纯前端下载 TXT"):
                trigger_frontend_download("ai_review.txt", "text/plain", st.session_state["ai_last_output"].encode("utf-8-sig"))
        with col_d2:
            if st.button("⬇ 纯前端下载 MD"):
                trigger_frontend_download("ai_review.md", "text/markdown", st.session_state["ai_last_output"].encode("utf-8-sig"))
    else:
        st.info("暂无结果。点击上方“开始生成”。")

    

    # ===============================
    # 4.1) 二次口令修改（迭代改写）
    # ===============================
    st.markdown("### 4.1) 根据口令修改本次综述（迭代改写）")

    revise_cmd = st.text_area(
        "修改口令/指令（例如：压缩到600字；增加局限性段；把结构改为IMRAD；突出临床证据；语气更学术等）",
        value="",
        height=110,
        key="review_revision_cmd"
    )

    if st.button("✍️ 按口令改写综述", key="btn_revision"):
        if not st.session_state.get("ai_last_output", "").strip():
            st.session_state["ai_notice"] = "当前没有可改写的综述，请先生成综述。"
        elif not revise_cmd.strip():
            st.session_state["ai_notice"] = "请输入修改口令后再改写。"
        else:
            cfg = load_ai_settings()
            base_url = cfg["base_url"].rstrip("/")
            api_key = cfg["api_key"]
            model = cfg["model"]
            temperature = cfg["temperature"]
            max_tokens = cfg["max_tokens"]
            system_prompt_custom = cfg["system_prompt"]

            # 关键：用“同一批文献”作为上下文，防止胡改
            # df_input 在本页面里已计算（load_articles_by_pmids(pmids_unique)）
            pmids_for_revision = st.session_state.get("ai_last_pmids") or list(pmids_unique)
            topic_for_revision = st.session_state.get("ai_last_topic_hint") or topic_hint
            df_rev = load_articles_by_pmids(pmids_for_revision) if pmids_for_revision else pd.DataFrame()

            if df_rev.empty:
                st.session_state["ai_notice"] = "缺少上一次生成综述对应的文献上下文，无法进行基于文献的改写。"
            else:
                sys_default, user_prompt2 = build_review_revision_prompts(
                    instruction=revise_cmd,
                    original_review=st.session_state["ai_last_output"],
                    df=df_rev,
                    topic_hint=topic_for_revision
                )
                system_prompt2 = sys_default + ("\n\n" + system_prompt_custom.strip() if system_prompt_custom.strip() else "")

                try:
                    with st.spinner("AI 正在按口令改写综述..."):
                        revised = call_chat_completions(
                            base_url=base_url,
                            api_key=api_key,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            system_prompt=system_prompt2,
                            user_prompt=user_prompt2
                        )

                    # 更新当前展示内容
                    st.session_state["ai_last_output"] = revised

                    parent_id = st.session_state.get("ai_last_review_id")
                    src = f"revision_of:{parent_id}" if parent_id else "revision"

                    # 保存“改写版”到 DB
                    new_id2 = save_ai_review_to_db(
                        source=src,
                        pmids=",".join([str(p) for p in pmids_unique]),
                        topic_hint=topic_hint,
                        base_url=base_url,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt2,
                        user_prompt=user_prompt2,
                        output=revised
                    )
                    st.session_state["ai_last_review_id"] = new_id2

                    st.success("已按口令改写，并保存为一条新的历史综述记录。")
                    st.rerun()
                except Exception as e:
                    st.session_state["ai_notice"] = f"AI 改写失败：{e}"

    st.markdown("### 5) 历史综述（来自本地数据库）")

    col_hclr, _ = st.columns([1, 4])
    if col_hclr.button("🗑 一键清空AI综述历史", key="clear_ai_reviews_btn"):
        st.session_state["confirm_clear_ai_reviews"] = True

    if st.session_state.get("confirm_clear_ai_reviews"):
        @st.dialog("确认清空AI综述历史？")
        def _confirm_clear_reviews():
            st.warning("将删除所有 AI 综述历史记录（ai_reviews），且不可恢复。")
            c1, c2 = st.columns(2)
            if c1.button("确认删除", key="confirm_clear_ai_reviews_yes"):
                clear_ai_reviews()
                st.session_state["ai_last_output"] = ""
                st.session_state["ai_last_review_id"] = None
                st.session_state["confirm_clear_ai_reviews"] = False
                st.rerun()
            if c2.button("取消", key="confirm_clear_ai_reviews_no"):
                st.session_state["confirm_clear_ai_reviews"] = False
                st.rerun()
        _confirm_clear_reviews()
    hist = list_ai_reviews(limit=50)
    if hist.empty:
        st.write("暂无历史记录。")
    else:
        options = hist.apply(lambda r: f"#{r['id']} | {r['created_at']} | {r['source']} | {r['model']} | {r['topic_hint']}", axis=1).tolist()
        sel = st.selectbox("选择一条历史记录", options, index=0)
        sel_id = int(sel.split("|")[0].strip().replace("#", ""))
        item = load_ai_review(sel_id)
        if item:
            with st.expander("查看详情", expanded=True):
                st.write(f"时间：{item['created_at']}")
                st.write(f"来源：{item['source']}")
                st.write(f"文献 PMID：{(item['pmids'] or '')[:200]}{'...' if item.get('pmids') and len(item['pmids'])>200 else ''}")
                st.write(f"模型：{item['model']} | temp={item['temperature']} | max_tokens={item['max_tokens']}")
                st.text_area("输出", item["output"] or "", height=320)
                col_del, col_h1, col_h2 = st.columns([1, 1, 1])
                with col_del:
                    if st.button("🗑 删除这条记录", key=f"del_ai_review_{sel_id}"):
                        st.session_state["confirm_delete_ai_review_id"] = sel_id

                if st.session_state.get("confirm_delete_ai_review_id") == sel_id:
                    @st.dialog("确认删除该条AI综述记录？")
                    def _confirm_delete_one_review():
                        st.warning(f"将删除 AI 综述记录 #{sel_id}，且不可恢复。")
                        c1, c2 = st.columns(2)
                        if c1.button("确认删除", key=f"confirm_del_ai_review_yes_{sel_id}"):
                            delete_ai_review(sel_id)
                            # 若删除的是当前“最新输出”对应记录，则同时清空当前输出
                            if st.session_state.get("ai_last_review_id") == sel_id:
                                st.session_state["ai_last_output"] = ""
                                st.session_state["ai_last_review_id"] = None
                            st.session_state["confirm_delete_ai_review_id"] = None
                            st.rerun()
                        if c2.button("取消", key=f"confirm_del_ai_review_no_{sel_id}"):
                            st.session_state["confirm_delete_ai_review_id"] = None
                            st.rerun()
                    _confirm_delete_one_review()

                with col_h1:
                    if st.button("⬇ 下载该条 TXT", key=f"dl_ai_review_txt_{sel_id}"):
                        trigger_frontend_download(f"ai_review_{sel_id}.txt", "text/plain", (item["output"] or "").encode("utf-8-sig"))
                with col_h2:
                    if st.button("⬇ 下载该条 MD", key=f"dl_ai_review_md_{sel_id}"):
                        trigger_frontend_download(f"ai_review_{sel_id}.md", "text/markdown", (item["output"] or "").encode("utf-8-sig"))


# ===============================
# 新页面：AI 对话（生成 PubMed 检索策略）+ 保存到DB
# ===============================
else:
    st.subheader("💬 AI 对话：生成 PubMed 文献检索策略（保存到本地数据库）")

    st.markdown("""
把你的研究问题/主题描述粘贴到下面（越具体越好：人群、干预/暴露、对照、结局、研究类型等）。  
点击生成后，AI 会输出：概念拆解 + 可复制的 PubMed 检索式 + 可选限制条件 + 追问问题。
""")

    user_text = st.text_area("你的描述", height=160, placeholder="例如：我想检索PD-1/PD-L1抑制剂在非小细胞肺癌一线治疗中的疗效与安全性...")

    col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
    with col_c1:
        if st.button("🧠 生成检索策略"):
            if not user_text.strip():
                st.session_state["ai_notice"] = "请输入一段描述后再生成。"
            else:
                cfg = load_ai_settings()
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
                                user_prompt=user_prompt
                            )
                        st.session_state["chat_last_output"] = out

                        # 保存到 DB
                        save_chat_log(
                            chat_type="pubmed_strategy",
                            base_url=base_url,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            system_prompt=system_prompt,
                            user_input=user_text,
                            assistant_output=out
                        )
                        st.success("已生成并保存到本地数据库（ai_chat_logs）")
                    except Exception as e:
                        st.session_state["ai_notice"] = f"AI 调用失败：{e}"

    with col_c2:
        if st.button("🧹 清空当前输出"):
            st.session_state["chat_last_output"] = ""

    with col_c3:
        st.caption("提示：可在侧边栏配置 Base URL / Key / Model 等（与 chatbox 一样），设置会保存到本地数据库。")

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

    col_hclr, _ = st.columns([1, 4])
    if col_hclr.button("🗑 一键清空AI检索策略历史", key="clear_pubmed_strategy_logs_btn"):
        st.session_state["confirm_clear_pubmed_strategy_logs"] = True

    if st.session_state.get("confirm_clear_pubmed_strategy_logs"):
        @st.dialog("确认清空AI检索策略历史？")
        def _confirm_clear_logs():
            st.warning("将删除所有“检索策略生成”AI历史记录（ai_chat_logs: pubmed_strategy），且不可恢复。")
            c1, c2 = st.columns(2)
            if c1.button("确认删除", key="confirm_clear_pubmed_strategy_logs_yes"):
                clear_chat_logs("pubmed_strategy")
                st.session_state["chat_last_output"] = ""
                st.session_state["confirm_clear_pubmed_strategy_logs"] = False
                st.rerun()
            if c2.button("取消", key="confirm_clear_pubmed_strategy_logs_no"):
                st.session_state["confirm_clear_pubmed_strategy_logs"] = False
                st.rerun()
        _confirm_clear_logs()
    hist = list_chat_logs("pubmed_strategy", limit=50)
    if hist.empty:
        st.write("暂无历史记录。")
    else:
        options = hist.apply(lambda r: f"#{r['id']} | {r['created_at']} | {r['model']} | {str(r['user_input'])[:40]}", axis=1).tolist()
        sel = st.selectbox("选择一条历史记录", options, index=0)
        sel_id = int(sel.split("|")[0].strip().replace("#", ""))
        item = load_chat_log(sel_id)
        if item:
            with st.expander("查看历史详情", expanded=True):
                st.write(f"时间：{item['created_at']}")
                st.write(f"模型：{item['model']} | temp={item['temperature']} | max_tokens={item['max_tokens']}")
                st.text_area("用户输入", item["user_input"] or "", height=140)
                st.text_area("AI 输出", item["assistant_output"] or "", height=320)

                col_del, col_h1, col_h2 = st.columns([1, 1, 1])
                with col_del:
                    if st.button("🗑 删除这条记录", key=f"del_pubmed_strategy_{sel_id}"):
                        st.session_state["confirm_delete_pubmed_strategy_id"] = sel_id

                if st.session_state.get("confirm_delete_pubmed_strategy_id") == sel_id:
                    @st.dialog("确认删除该条检索策略记录？")
                    def _confirm_delete_one_strategy():
                        st.warning(f"将删除检索策略记录 #{sel_id}，且不可恢复。")
                        c1, c2 = st.columns(2)
                        if c1.button("确认删除", key=f"confirm_del_pubmed_strategy_yes_{sel_id}"):
                            delete_chat_log(sel_id)
                            # 若当前页面输出来自该条记录，可同时清空
                            st.session_state["chat_last_output"] = ""
                            st.session_state["confirm_delete_pubmed_strategy_id"] = None
                            st.rerun()
                        if c2.button("取消", key=f"confirm_del_pubmed_strategy_no_{sel_id}"):
                            st.session_state["confirm_delete_pubmed_strategy_id"] = None
                            st.rerun()
                    _confirm_delete_one_strategy()

                with col_h1:
                    if st.button("⬇ 下载该条 TXT", key=f"dl_pubmed_strategy_txt_{sel_id}"):
                        trigger_frontend_download(f"pubmed_strategy_{sel_id}.txt", "text/plain", (item["assistant_output"] or "").encode("utf-8-sig"))
                with col_h2:
                    if st.button("⬇ 下载该条 MD", key=f"dl_pubmed_strategy_md_{sel_id}"):
                        trigger_frontend_download(f"pubmed_strategy_{sel_id}.md", "text/markdown", (item["assistant_output"] or "").encode("utf-8-sig"))









