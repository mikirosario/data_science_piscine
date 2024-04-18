from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import psycopg2
import glob
from psycopg2.extensions import connection, cursor
from psycopg2 import sql

# Database connection parameters
dbname = 'piscineds'
user = 'mrosario'
password = 'mysecretpassword'
host = 'localhost'
port = '5432'

# Customer directory location
customer_directory = '../customer/'

# Customer csv file paths
customer_csv_file_paths: list[str] = glob.glob(f"{customer_directory}*.csv")

# Max threads
max_threads = 6

def clear_db(conn: connection):
    with conn.cursor() as cur:
        # Query to select all table names in the current database's public schema
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        # Fetch all table names
        tables = cur.fetchall()
        # Generate and execute DROP TABLE commands for each table
        for table in tables:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table[0])))
        # Commit the changes
        conn.commit()

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

def load_csv_into_db(cur: cursor, csv_file_path: str, table_name: str):
    with open(csv_file_path, 'r') as csv_file:
        copy_sql = sql.SQL("""
            COPY {tname} FROM stdin WITH CSV HEADER DELIMITER as ','
        """).format(tname=sql.Identifier(table_name))
        cur.copy_expert(copy_sql, csv_file)

def load_csv_into_db_parallel(csv_file_path: str) -> str:
    try:
        # Extract the table name from the file path
        table_name = Path(csv_file_path).stem
        # Establish a new database connection for each thread
        with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
            with conn.cursor() as cur:
                # Assuming create_table_sql and load_csv_into_db are defined to handle cur, path, and table_name
                print(f"Generating table '{table_name}'...")
                create_table_sql(cur, table_name)
                conn.commit()  # Commit table creation before data loading
                print(f"Copying data into '{table_name}'...")
                load_csv_into_db(cur, csv_file_path, table_name)
                conn.commit()  # Commit after data loading
    except Exception as e:
        print(f"An error occurred while loading '{csv_file_path}': {e}")
    finally:
        if (conn):
            conn.close()
        return table_name

try:
    with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
        # Clear database
        clear_db(conn)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(load_csv_into_db_parallel, path) for path in customer_csv_file_paths]
        # Optionally, wait for all tasks to complete and handle results/errors
        for future in as_completed(futures):
            try:
                path = future.result()  # Exceptions thrown by the task will be re-raised here
                print(f"Copied {path}")
            except Exception as e:  # Now catching exceptions from the task
                print(f"Error processing file: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if conn:
        conn.close()
