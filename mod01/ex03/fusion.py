import psycopg2
from psycopg2.extensions import connection, cursor
from psycopg2 import sql

# Database connection parameters
dbname = 'piscineds'
user = 'mrosario'
password = 'mysecretpassword'
host = 'localhost'
port = '5432'

def drop_table_if_exists(cur: cursor, table_name: str):
    drop_sql = sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name))
    cur.execute(drop_sql)

try:
    with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
        with conn.cursor() as cur:
            drop_table_if_exists(cur, "customers_temp")
            conn.commit()  # Commit the changes to the database
            # Step 1: Fusionar tables en customers_temp
            cur.execute("""
                CREATE TABLE IF NOT EXISTS customers_temp AS
                SELECT 
                    c.event_time, c.event_type, c.product_id, c.price, c.user_id, c.user_session, 
                    string_agg(DISTINCT i.category_id::text, ', ') FILTER (WHERE i.category_id IS NOT NULL) AS category_id,
                    string_agg(DISTINCT i.category_code, ', ') FILTER (WHERE i.category_code IS NOT NULL) AS category_code,
                    string_agg(DISTINCT i.brand, ', ') FILTER (WHERE i.brand IS NOT NULL) AS brand
                FROM customers c
                LEFT JOIN item i ON c.product_id = i.product_id
                GROUP BY c.event_time, c.event_type, c.product_id, c.price, c.user_id, c.user_session;
            """)
            # Step 2: Swap the tables
            print("Backing up original 'customers' table and swapping...")
            cur.execute("DROP TABLE IF EXISTS customers_prejoin_backup;")
            cur.execute("ALTER TABLE customers RENAME TO customers_prejoin_backup;")
            cur.execute("ALTER TABLE customers_temp RENAME TO customers;")
            conn.commit()
            print("Table swap completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
