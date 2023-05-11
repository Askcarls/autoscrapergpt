import sqlite3

def setup_database():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback
        (id INTEGER PRIMARY KEY, 
        question TEXT, 
        answer TEXT, 
        feedback INTEGER)
    ''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    setup_database()