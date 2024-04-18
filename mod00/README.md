Para Python:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Symlinks:
ln -s "$HOME/goinfre/data" ../data
ln -s "$HOME/goinfre/customer" ../customer
ln -s "$HOME/goinfre/item" ../item

Para DB:
    docker-compose up
