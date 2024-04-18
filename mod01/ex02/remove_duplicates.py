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

def create_table_sql(cur: cursor, table_name: str):
    # SQL to create table
    table_sql = sql.SQL("""
    CREATE TABLE IF NOT EXISTS {tname} (
        event_time TIMESTAMP,
        event_type VARCHAR,
        product_id INT,
        price MONEY,
        user_id BIGINT,
        user_session UUID
    );
    """).format(tname=sql.Identifier(table_name))
    cur.execute(table_sql)

# # Create indices
# def create_indices_sql(cur: cursor):
#     # Dynamic parts of the index creation statements, such as index names and column names
#     indices_details = [
#         {
#             "index_name": "idx_event_time",
#             "table_name": "customers",
#             "columns": ["event_time"]
#         },
#         {
#             "index_name": "idx_cust_details",
#             "table_name": "customers",
#             "columns": ["event_type", "product_id", "price", "user_id", "user_session"]
#         }
#     ]
#     for index_detail in indices_details:
#         # Construct the CREATE INDEX SQL statement
#         index_sql = sql.SQL("CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns})").format(
#             index_name=sql.Identifier(index_detail["index_name"]),
#             table_name=sql.Identifier(index_detail["table_name"]),
#             columns=sql.SQL(", ").join(map(sql.Identifier, index_detail["columns"]))
#         )
#         cur.execute(index_sql)

# Establishing the database connection
try:
    with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
        print("Connected to the database.")
        with conn.cursor() as cur:
            drop_table_if_exists(cur, "customers_temp")
            # print("Indexing 'customers' table.")
            # create_indices_sql(cur)
            # conn.commit()
            # Step 1: Create a temporary table with the same structure
            # print("Indexing completed.")
            print("Creating temp 'customers' table...")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS customers_temp AS
            TABLE customers WITH NO DATA;
            """)
            conn.commit()  # Commit the changes to the database
            # Step 2: Populate the temporary table with deduplicated records
            print("Deduplicating 'customers' table...")
            # WHERE prev_event_time IS NULL OR EXTRACT(epoch FROM (event_time - prev_event_time)) > 1
            cur.execute("""
                INSERT INTO customers_temp
                SELECT
                    event_time, event_type, product_id, price, user_id, user_session
                FROM (
                    SELECT
                        *,
                        LAG(event_time) OVER (
                            PARTITION BY event_type, product_id, price, user_id, user_session
                            ORDER BY event_time
                        ) AS prev_event_time
                    FROM customers
                ) AS sorted_customers
                WHERE prev_event_time IS NULL OR event_time - prev_event_time > INTERVAL '1 second'
                ORDER BY event_time, event_type, product_id, price, user_id, user_session;
            """)
            conn.commit()  # Commit the changes to the database
            print("Transferring temporary table...")
            # Step 3: Swap the tables and back up original
            print("Backing up and swapping tables...")
            cur.execute("DROP TABLE IF EXISTS customers_prefiltered_backup;")
            cur.execute("ALTER TABLE customers RENAME TO customers_prefiltered_backup;")
            cur.execute("ALTER TABLE customers_temp RENAME TO customers;")
            # delete_duplicates_sql(cur)
            print("Deduplicating completed.")
except Exception as e:
    print(f"An error occurred while deleting duplicates: {e}")
# finally:
#     if conn:
#         conn.close()
