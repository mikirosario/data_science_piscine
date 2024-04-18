# cur.execute("""
#             SELECT
#                 COUNT(price::NUMERIC) AS total_count,
#                 AVG(price::NUMERIC) AS mean_price,
#                 STDDEV_POP(price::NUMERIC) AS stddev_price,
#                 MIN(price::NUMERIC) AS min_price,
#                 PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price::NUMERIC) AS first_quartile,
#                 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price::NUMERIC) AS second_quartile,
#                 PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price::NUMERIC) AS third_quartile,
#                 MAX(price::NUMERIC) AS max_price,
#                 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price::NUMERIC) AS median_price,
#             FROM customers
#             WHERE event_type = 'purchase';
#             """)
# Fetch all purchase prices
import psycopg2
import plotly.graph_objects as go
import numpy as np

# Database connection parameters
dbname = 'piscineds'
user = 'mrosario'
password = 'mysecretpassword'
host = 'localhost'
port = '5432'

def show_box_plot(prices: np.ndarray):
    # Horizontal Box Plot of All Data
    fig = go.Figure()
    fig.add_trace(go.Box(x=prices, name="Purchase Price Distribution", orientation='h'))
    fig.update_layout(
        xaxis_title="price",
        yaxis_title=None
    )
    fig.show()

def show_iqt_box_plot(prices: np.ndarray):
    # Horizontal Box Plot Focused Between the First and Third Quartiles
    # Filtering prices to include only those within the interquartile range:
    # 1. Generate boolean arrays for all prices (prices >= q1) and (prices <= q3) where each element is set to True if the condition is met.
    # 2. Combine arrays with bitwise AND to getsingle boolean array for all prices where True indicates corresponding indexed price is >= q1 or <= q3.
    # 3. Passing the boolean array as an indexer into a numpy array causes only the prices corresponding to the True indices to be selected. Python. xD
    filtered_prices: np.ndarray = prices[(prices >= q1) & (prices <= q3)]
    fig = go.Figure()
    fig.add_trace(go.Box(x=filtered_prices, name="Interquartile Price Range", orientation='h'))
    fig.update_layout(
        xaxis_title="price",
        yaxis_title=None
    )
    fig.show()

def show_avg_bskt_price_box_plot(basket_prices: np.ndarray):
    fig = go.Figure()
    # Calculate the first and third quartiles (Q1 and Q3)
    q1, q3 = np.percentile(basket_prices, [25, 75])
    iqr = q3 - q1
    whisker_factor = 1.5
    lower_whisker = q1 - whisker_factor * iqr
    upper_whisker = q3 + whisker_factor * iqr
    # Filter basket prices to get only those within the interquartile range
    basket_prices = basket_prices[(basket_prices >= lower_whisker) & (basket_prices <= upper_whisker)]
    fig.add_trace(go.Box(x=basket_prices, name="Average Basket Price", boxpoints='outliers', orientation='h'))
    fig.update_layout(
        title="Average Basket Price per User Session",
        yaxis_title="Average Basket Price ($)"
    )
    fig.show()

def get_all_prices_from_db() -> np.ndarray:
    with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
        with conn.cursor() as cur:            
            cur.execute("""
                        SELECT
                            price::NUMERIC
                        FROM customers
                        WHERE event_type = 'purchase';
                        """)
            prices: np.ndarray = np.array([float(record[0]) for record in cur.fetchall()])
    return prices

def get_avg_basket_price_per_user_from_db() -> np.ndarray:
    with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                        SELECT user_id, AVG(total_price) AS avg_basket_price
                        FROM (
                            SELECT user_id, user_session, SUM(price::NUMERIC) AS total_price
                            FROM customers
                            WHERE event_type = 'purchase'
                            GROUP BY user_id, user_session
                        ) AS session_totals
                        GROUP BY user_id;
                        """)  # This gives you the average basket price per session per user
            basket_prices = np.array([float(record[1]) for record in cur.fetchall()])
    return basket_prices

try:
    prices: np.ndarray = get_all_prices_from_db()
    basket_prices: np.ndarray = get_avg_basket_price_per_user_from_db()
    mean_price = np.mean(prices)
    std_dev_price = np.std(prices)
    min_price = np.min(prices)
    max_price = np.max(prices)
    q1 = np.percentile(prices, 25)
    q2 = np.percentile(prices, 50)
    # median_price = np.median(prices)
    q3 = np.percentile(prices, 75)
    total_count = prices.size

    # Print Data
    print(f"count: {total_count}")
    print(f"mean: {mean_price:.5f}")
    print(f"std: {std_dev_price:.5f}")
    print(f"min: {min_price:.5f}")
    print(f"25%: {q1:.5f}")
    print(f"50% (median): {q2:.5f}")
    print(f"75%: {q3:.5f}")
    print(f"max: {max_price:.5f}")

    # Show box plots
    show_box_plot(prices)
    show_iqt_box_plot(prices)
    show_avg_bskt_price_box_plot(basket_prices)
except Exception as e:
    print(f"An error occurred: {e}")
