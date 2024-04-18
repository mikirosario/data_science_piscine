import psycopg2
from psycopg2.extensions import cursor
from psycopg2 import sql

# Database connection parameters
dbname = 'piscineds'
user = 'mrosario'
password = 'mysecretpassword'
host = 'localhost'
port = '5432'

# Customers table name
customers_table_name = "customers"

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

def find_tables_to_union(cur: cursor) -> list[str]:
    # Query to find tables starting with 'data_202'
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name LIKE 'data_202%%'
    """)
    return [row[0] for row in cur.fetchall()] # cur.fetchall returns a list of tuples with keys indicated in select, applying constraints in WHERE statement

def build_union_all_sql(source_table_names: list[str], target_table_name: str):
    # Base SELECT statement for UNION ALL; assuming all tables have the same structure
    select_statements = [
        sql.SQL("SELECT event_time, event_type, product_id, price, user_id, user_session FROM {}")
        .format(sql.Identifier(table_name)) for table_name in source_table_names
    ]
    # Combine all SELECT statements with UNION ALL
    combined_statements = sql.SQL(" UNION ALL ").join(select_statements)
    # Construct the full INSERT INTO ... UNION ALL ... command
    full_sql = sql.SQL("INSERT INTO {} ").format(sql.Identifier(target_table_name)) + combined_statements
    return full_sql

def union_tables_into_customers(cur: cursor, source_table_names: list[str], target_table_name: str):
    union_sql = build_union_all_sql(source_table_names, target_table_name)
    cur.execute(union_sql)

try:
    # Establish a new database connection for each thread
    with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
        with conn.cursor() as cur:
            tables_to_union: list[str] = find_tables_to_union(cur)
            if (len(tables_to_union) < 1):
                raise Exception("No tables '202*' found to union in database")

            # Drop the 'customers' table if it exists (for testing purposes since this script will be run multiple times)
            print(f"Dropping '{customers_table_name}' table if it exists...")
            drop_table_if_exists(cur, customers_table_name)
            # Commit the DROP TABLE operation
            conn.commit()
            
            # Create the target 'customers' table if it does not exist
            print(f"Generating customer table '{customers_table_name}'...")
            create_table_sql(cur, customers_table_name)
            conn.commit()

            # Perform the UNION ALL operation and insert data into 'customers'
            print(f"Unioning data into '{customers_table_name}' table...")
            union_tables_into_customers(cur, tables_to_union, customers_table_name)
            conn.commit()  # Commit after union operation
            
except Exception as e:
    print(f"An error occurred while creating the table '{customers_table_name}': {e}")
finally:
    if conn:
        conn.close()
