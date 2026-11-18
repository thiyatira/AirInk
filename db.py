"""AirInk's MySQL connection: the same setup as the other Tkinter + MySQL projects."""

import os

import mysql.connector

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "airink_db"),
}
Error = mysql.connector.Error


def init_db():
    """First run: if the database doesn't exist yet, create it from schema.sql."""
    server = {k: v for k, v in DB_CONFIG.items() if k != "database"}
    conn = mysql.connector.connect(connection_timeout=3, **server)
    try:
        cur = conn.cursor()
        cur.execute("SHOW DATABASES LIKE %s", (DB_CONFIG["database"],))
        if cur.fetchone():
            return  # already set up, keep existing data
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql"), encoding="utf-8") as f:
            for statement in f.read().split(";"):
                statement = statement.strip().replace("airink_db", DB_CONFIG["database"])
                if statement:
                    cur.execute(statement)
        conn.commit()
    finally:
        conn.close()


def run(sql, params=(), fetch=False):
    """Execute one query. Returns rows when fetch=True, otherwise the new row's id."""
    conn = mysql.connector.connect(connection_timeout=3, **DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        if fetch:
            return cur.fetchall()
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()
