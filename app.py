import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import sqlite3
import base64
import time
import os
import hashlib
import secrets
from datetime import datetime

# ===============================
# 全局路径设置
# ===============================

USERS_DB = "users.db"
USER_DB_DIR = "user_dbs"


# ===============================
# 用户系统
# ===============================

def _ensure_dirs():
    os.makedirs(USER_DB_DIR, exist_ok=True)


def init_users_db():
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            pwd_salt TEXT NOT NULL,
            pwd_hash TEXT NOT NULL,
            security_question TEXT NOT NULL,
            sec_salt TEXT NOT NULL,
            sec_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _pbkdf2_hash(text: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", text.encode("utf-8"), salt, 120000)
    return dk.hex()


def _new_salt_hex():
    return secrets.token_bytes(16).hex()


def create_user(username, password, question, answer):
    if not username or len(username) < 3:
        raise ValueError("用户名至少3个字符")
    if not password or len(password) < 6:
        raise ValueError("密码至少6位")
    if not question.strip():
        raise ValueError("密保问题不能为空")
    if not answer.strip():
        raise ValueError("密保答案不能为空")

    pwd_salt = _new_salt_hex()
    pwd_hash = _pbkdf2_hash(password, pwd_salt)

    sec_salt = _new_salt_hex()
    sec_hash = _pbkdf2_hash(answer.lower().strip(), sec_salt)

    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            username,
            pwd_salt,
            pwd_hash,
            question,
            sec_salt,
            sec_hash,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError("用户名已存在")
    finally:
        conn.close()


def verify_login(username, password):
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("SELECT pwd_salt, pwd_hash FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return False
    return _pbkdf2_hash(password, row[0]) == row[1]


def get_security_question(username):
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("SELECT security_question FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def reset_password(username, answer, new_password):
    if len(new_password) < 6:
        raise ValueError("新密码至少6位")

    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("SELECT sec_salt, sec_hash FROM users WHERE username=?", (username,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise ValueError("用户不存在")

    if _pbkdf2_hash(answer.lower().strip(), row[0]) != row[1]:
        conn.close()
        raise ValueError("密保答案错误")

    new_salt = _new_salt_hex()
    new_hash = _pbkdf2_hash(new_password, new_salt)

    c.execute("UPDATE users SET pwd_salt=?, pwd_hash=? WHERE username=?",
              (new_salt, new_hash, username))
    conn.commit()
    conn.close()


def user_db_path(username):
    _ensure_dirs()
    h = hashlib.sha256(username.encode()).hexdigest()[:16]
    return os.path.join(USER_DB_DIR, f"user_{h}.db")


def get_db_file():
    if "db_file" not in st.session_state or not st.session_state["db_file"]:
        raise RuntimeError("未登录")
    return st.session_state["db_file"]
# ===============================
# 每个用户的文章数据库初始化
# ===============================

def init_db(db_file):
    conn = sqlite3.connect(db_file)
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
            pmid TEXT PRIMARY KEY
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            idx INTEGER PRIMARY KEY,
            pmid TEXT UNIQUE
        )
    """)

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

    c.execute("""
        CREATE TABLE IF NOT EXISTS ai_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            output TEXT
        )
    """)

    conn.commit()
    conn.close()


# ===============================
# 搜索缓存
# ===============================

def clear_search_cache():
    conn = sqlite3.connect(get_db_file())
    c = conn.cursor()
    c.execute("DELETE FROM search_cache")
    conn.commit()
    conn.close()


def save_search_results_to_db(articles):
    conn = sqlite3.connect(get_db_file())
    c = conn.cursor()

    c.execute("DELETE FROM search_cache")

    for i, article in enumerate(articles):
        c.execute("""
            INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            article.get("pmid"),
            article.get("title"),
            article.get("journal"),
            article.get("year"),
            article.get("abstract"),
            article.get("doi"),
            article.get("pmcid"),
        ))

        c.execute("""
            INSERT INTO search_cache VALUES (?, ?)
        """, (i, article.get("pmid")))

    conn.commit()
    conn.close()


def load_search_page(page, page_size):
    offset = page * page_size
    conn = sqlite3.connect(get_db_file())
    df = pd.read_sql_query("""
        SELECT a.*
        FROM search_cache s
        JOIN articles a ON a.pmid = s.pmid
        ORDER BY s.idx
        LIMIT ? OFFSET ?
    """, conn, params=(page_size, offset))
    conn.close()
    return df


def get_search_total_count():
    conn = sqlite3.connect(get_db_file())
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM search_cache")
    total = c.fetchone()[0]
    conn.close()
    return total


# ===============================
# 收藏功能
# ===============================

def add_favorite(pmid):
    conn = sqlite3.connect(get_db_file())
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO favorites VALUES (?)", (pmid,))
    conn.commit()
    conn.close()


def remove_favorite(pmid):
    conn = sqlite3.connect(get_db_file())
    c = conn.cursor()
    c.execute("DELETE FROM favorites WHERE pmid=?", (pmid,))
    conn.commit()
    conn.close()


def load_favorites():
    conn = sqlite3.connect(get_db_file())
    df = pd.read_sql_query("""
        SELECT a.*
        FROM articles a
        JOIN favorites f ON a.pmid = f.pmid
    """, conn)
    conn.close()
    return df


# ===============================
# PubMed 搜索
# ===============================

def search_pubmed(query, retmax=20):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search = requests.get(
        base + "esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json"},
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
            "abstract": " ".join(
                [a.text for a in article.findall(".//AbstractText") if a.text]
            ),
            "doi": None,
            "pmcid": None
        })

    return articles

# ===============================
# Streamlit UI
# ===============================

st.set_page_config(layout="wide")
st.title("📚 PubMed 文献检索系统（多用户版）")

# 初始化用户数据库（仅用户表）
init_users_db()

# session 初始化
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "db_file" not in st.session_state:
    st.session_state["db_file"] = None
if "page" not in st.session_state:
    st.session_state["page"] = 0


def after_login(username):
    st.session_state["logged_in"] = True
    st.session_state["username"] = username
    st.session_state["db_file"] = user_db_path(username)

    # 登录成功后才初始化数据库
    init_db(st.session_state["db_file"])
    clear_search_cache()

    st.session_state["page"] = 0


# ===============================
# 未登录界面
# ===============================

if not st.session_state["logged_in"]:

    tab1, tab2, tab3 = st.tabs(["登录", "注册", "找回密码"])

    with tab1:
        u = st.text_input("用户名")
        p = st.text_input("密码", type="password")
        if st.button("登录"):
            if verify_login(u, p):
                after_login(u)
                st.success("登录成功")
                st.rerun()
            else:
                st.error("用户名或密码错误")

    with tab2:
        ru = st.text_input("新用户名")
        rp = st.text_input("新密码", type="password")
        rq = st.text_input("密保问题")
        ra = st.text_input("密保答案", type="password")
        if st.button("注册"):
            try:
                create_user(ru, rp, rq, ra)
                st.success("注册成功，请登录")
            except Exception as e:
                st.error(str(e))

    with tab3:
        fu = st.text_input("用户名")
        q = get_security_question(fu)
        if q:
            st.info(f"密保问题：{q}")
            ans = st.text_input("密保答案", type="password")
            np = st.text_input("新密码", type="password")
            if st.button("重置密码"):
                try:
                    reset_password(fu, ans, np)
                    st.success("密码已重置")
                except Exception as e:
                    st.error(str(e))

    st.stop()


# ===============================
# 登录后界面
# ===============================

st.sidebar.markdown(f"👤 当前用户：{st.session_state['username']}")
if st.sidebar.button("退出登录"):
    for k in ["logged_in", "username", "db_file"]:
        st.session_state.pop(k, None)
    st.rerun()

page = st.sidebar.radio("页面", ["🔍 文献检索", "📌 我的收藏"])

# ===============================
# 文献检索页面
# ===============================

if page == "🔍 文献检索":

    query = st.text_input("关键词", "cancer immunotherapy")
    retmax = st.slider("返回数量", 1, 100, 20)

    if st.button("搜索"):
        results = search_pubmed(query, retmax)
        save_search_results_to_db(results)
        st.session_state["page"] = 0
        st.rerun()

    total = get_search_total_count()
    if total > 0:

        page_size = 10
        total_pages = (total - 1) // page_size + 1
        current = st.session_state["page"]

        col1, col2 = st.columns(2)
        if col1.button("上一页", disabled=current == 0):
            st.session_state["page"] -= 1
            st.rerun()
        if col2.button("下一页", disabled=current >= total_pages - 1):
            st.session_state["page"] += 1
            st.rerun()

        df = load_search_page(current, page_size)

        for _, row in df.iterrows():
            st.markdown(f"### {row['title']}")
            st.markdown(f"期刊: {row['journal']} | 年份: {row['year']}")
            st.markdown(f"PMID: {row['pmid']}")
            if st.button("⭐ 收藏", key=f"fav_{row['pmid']}"):
                add_favorite(row["pmid"])
                st.success("已收藏")
            st.markdown("---")

    else:
        st.info("暂无搜索结果")


# ===============================
# 收藏页面
# ===============================

elif page == "📌 我的收藏":

    fav_df = load_favorites()

    if fav_df.empty:
        st.info("暂无收藏")
    else:
        for _, row in fav_df.iterrows():
            col1, col2 = st.columns([10, 1])
            col1.markdown(f"{row['title']}")
            if col2.button("❌", key=f"del_{row['pmid']}"):
                remove_favorite(row["pmid"])
                st.rerun()
