from database import get_db

conn = get_db()
cur = conn.cursor()

cur.execute("""
SELECT id, section_title, LEFT(content, 250)
FROM handbook_chunks
WHERE is_active = 1
  AND (section_title LIKE '%First-Degree%'
    OR section_title LIKE '%Core Values%'
    OR section_title LIKE '%Second-Degree%')
ORDER BY id DESC
LIMIT 8
""")

for r in cur.fetchall():
    print('---')
    print(r[0], '|', r[1])
    print(r[2])
    print()

cur.close()
conn.close()