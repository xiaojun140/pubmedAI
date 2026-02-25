# app_login.py

import streamlit as st
import sqlite3
import hashlib
import os
from datetime import datetime

# ----------------------------
# 配置
# ----------------------------
DB_FOLDER = "user_dbs"
os.makedirs(DB_FOLDER, exist_ok=True)
AUTH_DB = "users_auth.db"

# ----------------------------
# 工具函数
# ----------------------------

def hash_username(username):
    """返回用户名 hash 用于生成用户数据库文件名"""
    return hashlib.sha256(username.encode()).hexdigest()[:10]

def get_user_db(username):
    """返回用户数据库路径"""
    return os.path.join(DB_FOLDER, f"user_{hash_username(username)}.db")

def init_user_db(db_file):
    """初始化用户数据库结构，保持与 app_function.py 兼容"""
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # articles 表
    c.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            created_at TEXT
        )
    """)

    # 收藏表
    c.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            created_at TEXT
        )
    """)

    # 搜索缓存
    c.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            results TEXT,
            created_at TEXT
        )
    """)

    # AI 设置
    c.execute("""
        CREATE TABLE IF NOT EXISTS ai_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            temperature REAL,
            max_tokens INTEGER,
            system_prompt TEXT
        )
    """)

    # AI 评论
    c.execute("""
        CREATE TABLE IF NOT EXISTS ai_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            review TEXT,
            created_at TEXT
        )
    """)

    # AI 聊天日志
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
    conn.commit()
    conn.close()

# ----------------------------
# 用户认证
# ----------------------------

def register_user(username, password):
    """注册用户"""
    conn = sqlite3.connect(AUTH_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False  # 用户已存在
    conn.close()
    # 注册成功后创建用户独立数据库
    init_user_db(get_user_db(username))
    return True

def login_user(username, password):
    """登录用户"""
    conn = sqlite3.connect(AUTH_DB)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row and row[0] == password:
        st.session_state["username"] = username
        st.session_state["db_file"] = get_user_db(username)
        return True
    return False

# ----------------------------
# Streamlit 页面
# ----------------------------

def main():
    st.title("🔐 登录 / 注册")

    if "username" not in st.session_state:
        st.session_state["username"] = None

    if st.session_state["username"] is None:
        tab1, tab2 = st.tabs(["登录", "注册"])

        with tab1:
            username = st.text_input("用户名", key="login_user")
            password = st.text_input("密码", type="password", key="login_pass")
            if st.button("登录"):
                if login_user(username, password):
                    st.success(f"登录成功！欢迎 {username}")
                    st.experimental_rerun()
                else:
                    st.error("用户名或密码错误")

        with tab2:
            reg_user = st.text_input("用户名", key="reg_user")
            reg_pass = st.text_input("密码", type="password", key="reg_pass")
            if st.button("注册"):
                if register_user(reg_user, reg_pass):
                    st.success(f"注册成功！请登录 {reg_user}")
                else:
                    st.error("用户已存在")
    else:
        st.success(f"已登录: {st.session_state['username']}")
        st.info("请前往功能页面使用 AI 文章管理功能")
        if st.button("登出"):
            st.session_state["username"] = None
            st.session_state["db_file"] = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()
