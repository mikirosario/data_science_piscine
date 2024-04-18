import plotly.graph_objects as go
from datetime import datetime
import psycopg2
from psycopg2.extensions import connection, cursor
from psycopg2 import extensions

# Database connection parameters
dbname = 'piscineds'
user = 'mrosario'
password = 'mysecretpassword'
host = 'localhost'
port = '5432'

def show_line_chart():
    print("Connecting to database...")
    with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
        with conn.cursor() as cur:
            print("Fetching data...")
            cur.execute("""
                        SELECT
                            event_time::date AS day,
                            COUNT(*) AS purchase_count
                        FROM customers
                        WHERE event_type = 'purchase'
                        GROUP BY day
                        ORDER BY day;
                        """)
            results: list[tuple] = cur.fetchall()
    print("Generating line chart...")
    days = [result[0] for result in results]  # Dates
    purchase_counts = [result[1] for result in results]  # Purchase counts
    # Determine the first day of each month in the data range for x-axis ticks
    first_days_of_months = [day.strftime('%Y-%m-01') for day in set(datetime(day.year, day.month, 1) for day in days)]
    first_days_of_months.sort()
    # Pie chart
    fig = go.Figure()
    fig = go.Figure(data=go.Scatter(x=days, y=purchase_counts, mode='lines+markers', name='Daily Purchases'))
    fig.update_layout(xaxis_title='Date',
                    yaxis_title='Number of Customers',
                        xaxis=dict(
                            tickvals=first_days_of_months,  # Set tick values to the first day of each month
                            tickformat="%b %Y",  # Format as abbreviated month and full year
                        ),
                        yaxis=dict(
                            tickformat=".0f",  # Format as whole number with thousands separator
                        )
    )
    fig.show()

def show_bar_chart():
    print("Connecting to database...")
    with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
        with conn.cursor() as cur:
            print("Fetching data...")
            cur.execute("""
                        SELECT
                            DATE_TRUNC('month', event_time) AS month,
                            SUM(CAST(price AS NUMERIC)) AS total_revenue
                        FROM customers
                        WHERE event_type = 'purchase'
                        GROUP BY month
                        ORDER BY month;
                        """)
            results = cur.fetchall()
    print("Generating bar chart...")
    months = [result[0].strftime("%B %Y") for result in results]  # Format month names
    total_revenue_millions = [float(result[1]) / 1e6 for result in results]  # Convert to millions and ensure numeric type

    # Create the vertical bar chart
    fig = go.Figure(data=go.Bar(y=total_revenue_millions, x=months, orientation='v'))
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Total Sales in Millions of ₳',
        xaxis=dict(
            tickformat=".1f",  # Format x-axis ticks with one decimal place
        )
    )
    fig.show()

def show_area_chart():
    print("Connecting to database...")
    with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
        with conn.cursor() as cur:
            print("Fetching data...")
            cur.execute("""
                SELECT
                    event_time::date AS day,
                    AVG(spend::NUMERIC) AS average_spend
                FROM (
                    SELECT
                        event_time,
                        user_id,
                        SUM(price) AS spend
                    FROM customers
                    WHERE event_type = 'purchase'
                    GROUP BY event_time, user_id
                ) AS daily_spend
                GROUP BY day
                ORDER BY day;
                """)
            results = cur.fetchall()
    days = [result[0] for result in results]
    average_spends = [result[1] for result in results]
    # Determine the first day of each month in the data range for x-axis ticks
    first_days_of_months = [day.strftime('%Y-%m-01') for day in set(datetime(day.year, day.month, 1) for day in days)]
    first_days_of_months.sort()
    # Create the area chart
    fig = go.Figure(go.Scatter(x=days, y=average_spends, fill='tozeroy'))  # 'tozeroy' fills the area under the line to the y=0
    fig.update_layout(
        title='Average Daily Spend Per Customer',
        xaxis_title='Date',
        yaxis_title='Average Spend/Customers in ₳',
        xaxis=dict(
            tickvals=first_days_of_months,  # Set tick values to the first day of each month
            tickformat="%b %Y",  # Format as abbreviated month and full year
        )
    )
    fig.show()
try:
    show_line_chart()
    show_bar_chart()
    show_area_chart()
except Exception as e:
    print(f"An error occurred: {e}")
