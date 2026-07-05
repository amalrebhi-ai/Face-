import sqlite3

conn = sqlite3.connect("face_recognition.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    info TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS encodings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name TEXT,
    encoding BLOB
)
""")

conn.commit()
import pickle

def add_person(name, info, encoding):
    conn = sqlite3.connect("face_recognition.db")
    cursor = conn.cursor()

    # Ajouter personne
    cursor.execute(
        "INSERT OR IGNORE INTO persons (name, info) VALUES (?, ?)",
        (name, "\n".join(info))
    )

    # Sauvegarder encoding (converti en bytes)
    encoding_blob = pickle.dumps(encoding)

    cursor.execute(
        "INSERT INTO encodings (person_name, encoding) VALUES (?, ?)",
        (name, encoding_blob)
    )

    conn.commit()
    conn.close()