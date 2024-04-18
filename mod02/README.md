Para Python:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Symlinks:
ln -s "$HOME/goinfre/data" data
ln -s "$HOME/goinfre/customer" customer
ln -s "$HOME/goinfre/item" item

Para DB:
    docker-compose up

Query to order rows by event_time in pgAdmin
    SELECT * FROM public.customers
    ORDER BY event_time DESC
    LIMIT 100;

Remove duplicates test:
    Diff
    Org     -       New     = Duplicates
    20.692.840 - 17.062.104 = 3.630.736

    Ejemplo:
    En original:
    61 2022-12-01 00:03:08,remove_from_cart,5859482,$1.60,561162056,39cf2227-03ed-421e-9615-7814b9b3c5e6
    62 2022-12-01 00:03:08,remove_from_cart,5859482,$1.60,561162056,39cf2227-03ed-421e-9615-7814b9b3c5e6

    SELECT * FROM public.customers
    WHERE user_session = '39cf2227-03ed-421e-9615-7814b9b3c5e6'
    ORDER BY event_time;

    SELECT * FROM public.customers_prefiltered_backup
    WHERE user_session = '39cf2227-03ed-421e-9615-7814b9b3c5e6'
    ORDER BY event_time;
