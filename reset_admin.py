# reset_admin.py
from werkzeug.security import generate_password_hash
import pymysql

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'db': 'school_chatbot',
    'port': 3306,
}


def reset_admin():
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Delete existing admin
    cur.execute("DELETE FROM admin_users WHERE username='admin212004'")

    # Create new admin with password 'admin123'
    hashed = generate_password_hash('admin123')
    cur.execute(
        "INSERT INTO admin_users (username, password) VALUES (%s, %s)",
        ('admin212004', hashed)
    )
    conn.commit()

    print("✅ Admin user reset successfully!")
    print("   Username: admin212004")
    print("   Password: admin123")

    # Verify
    cur.execute("SELECT username FROM admin_users WHERE username='admin212004'")
    if cur.fetchone():
        print("✅ Verification: Admin exists in database")
    else:
        print("❌ Verification: Admin not found!")

    cur.close()
    conn.close()


if __name__ == "__main__":
    reset_admin()