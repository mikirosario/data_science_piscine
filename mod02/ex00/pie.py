import plotly.graph_objects as go
import psycopg2
from psycopg2.extensions import connection, cursor
from psycopg2 import sql

# Database connection parameters
dbname = 'piscineds'
user = 'mrosario'
password = 'mysecretpassword'
host = 'localhost'
port = '5432'

try:
    with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                        SELECT event_type, COUNT(*) as event_count
                        FROM customers
                        GROUP BY event_type;
                        """)
            results: list[tuple] = cur.fetchall()
    event_types = [result[0] for result in results]
    event_counts = [result[1] for result in results]
    # Pie chart
    fig = go.Figure(data=[go.Pie(labels=event_types, values=event_counts, textinfo='label+percent', textposition='inside')])
    fig.update_layout(showlegend=False)
    fig.show()
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if conn:
        conn.close()
    if cur:
        cur.close()
