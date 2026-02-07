import pyodbc

try:
    # Using Windows Authentication (no sa password needed)
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost,1433;'
        'Trusted_Connection=yes;'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT @@VERSION")
    version = cursor.fetchone()
    print("✓ Connection successful!")
    print(f"SQL Server version: {version[0]}")
    conn.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")