
import sqlite3
import bcrypt
import json
from datetime import datetime

DATABASE_NAME = "users.db"

def init_db():
    """Initializes the SQLite database for users and conversations."""
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()

    # Create users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Create conversations table
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database '{DATABASE_NAME}' initialized.")

def register_user(username, password):
    """Registers a new user with a hashed password. Returns True on success, False if user exists."""
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password.decode('utf-8')))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Username already exists
        return False
    finally:
        conn.close()

def verify_user(username, password):
    """Verifies user credentials. Returns user_id on success, None otherwise."""
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        user_id, hashed_password = result
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            return user_id
    return None

def save_conversation(user_id, question, answer):
    """Saves a question and its answer to the conversation history for a given user."""
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute(
        "INSERT INTO conversations (user_id, timestamp, question, answer) VALUES (?, ?, ?, ?)",
        (user_id, timestamp, question, answer)
    )
    conn.commit()
    conn.close()

def get_conversations(user_id):
    """Retrieves all conversation history for a given user."""
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("SELECT timestamp, question, answer FROM conversations WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    conversations = c.fetchall()
    conn.close()
    return conversations

# Initialize the database when the module is imported (or on first run)
init_db()
