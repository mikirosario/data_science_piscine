import psycopg2
import plotly.graph_objects as go
import numpy as np

# Database connection parameters
dbname = 'piscineds'
user = 'mrosario'
password = 'mysecretpassword'
host = 'localhost'
port = '5432'

# Purchase per user session
# SELECT user_id, COUNT(DISTINCT user_session) AS order_count
# FROM customers
# WHERE event_type = 'purchase'
# GROUP BY user_id;

def get_customer_orders_from_db() -> list[tuple]:
    with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
        with conn.cursor() as cur:            
            cur.execute("""
            SELECT user_id, COUNT(*) AS order_count
            FROM customers
            WHERE event_type = 'purchase'
            GROUP BY user_id;
            """)
            customer_orders = cur.fetchall()
    return customer_orders

def get_customer_spends_from_db() -> list[tuple]:
    with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
        with conn.cursor() as cur:            
            cur.execute("""
            SELECT SUM(price::NUMERIC) AS total_spend
            FROM customers
            WHERE event_type = 'purchase'
            GROUP BY user_id;
            """)
            customer_spends = cur.fetchall()
    return customer_spends

def show_order_frequency_bar_chart(order_counts):
    order_bins = np.arange(1, 40, 7)
    customer_order_counts, _ = np.histogram(order_counts, bins=order_bins)
    # Bin labels for the x-axis (e.g., "0-10", "10-20", ...)
    order_bin_labels = ['{} - {}'.format(order_bins[i], order_bins[i+1]) for i in range(len(order_bins)-1)]
    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(x=order_bin_labels, y=customer_order_counts)
    ])
    fig.update_layout(
        title='Number of Customers by Order Count Intervals',
        xaxis_title='Order Count Interval',
        yaxis_title='Number of Customers',
        xaxis={'type': 'category'}
    )
    fig.show()

def show_spend_frequency_bar_chart(customer_spends):
    spend_bins = list(range(-26, 275, 49))
    customer_counts, _ = np.histogram(customer_spends, bins=spend_bins)
    # Bin labels for the x-axis (e.g., "0-50", "50-100", ...)
    spend_bin_labels = ['{} - {}'.format(spend_bins[i], spend_bins[i+1]) for i in range(len(spend_bins)-1)]
    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(x=spend_bin_labels, y=customer_counts)
    ])
    fig.update_layout(
        title='Number of Customers by Total Spend Intervals',
        xaxis_title='Total Spend Interval ($)',
        yaxis_title='Number of Customers',
        xaxis={'type': 'category'}
    )
    fig.show()

try:
    customer_orders = get_customer_orders_from_db()
    show_order_frequency_bar_chart(customer_orders)
    customer_spends = get_customer_spends_from_db()
    show_spend_frequency_bar_chart(customer_spends)
except Exception as e:
    print(f"An error occurred: {e}")
