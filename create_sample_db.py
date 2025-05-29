import sqlite3

def create_sample_db(db_path="mydata.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop tables if exist for clean slate
    cursor.execute("DROP TABLE IF EXISTS customers;")
    cursor.execute("DROP TABLE IF EXISTS orders;")
    cursor.execute("DROP TABLE IF EXISTS products;")

    # Create customers table
    cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            country TEXT NOT NULL,
            email TEXT
        );
    """)

    # Create products table
    cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            price REAL NOT NULL
        );
    """)

    # Create orders table
    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY(product_id) REFERENCES products(product_id)
        );
    """)

    # Insert sample customers
    cursor.executemany("INSERT INTO customers (customer_name, country, email) VALUES (?, ?, ?);", [
        ("Alice Johnson", "USA", "alice@example.com"),
        ("Bob Smith", "Germany", "bob.smith@example.de"),
        ("Charlie Lee", "USA", "charlie.lee@example.com"),
        ("Diana Prince", "France", "diana.prince@example.fr")
    ])

    # Insert sample products
    cursor.executemany("INSERT INTO products (product_name, price) VALUES (?, ?);", [
        ("Laptop", 1200.00),
        ("Smartphone", 800.00),
        ("Tablet", 400.00),
        ("Headphones", 150.00)
    ])

    # Insert sample orders
    cursor.executemany("INSERT INTO orders (customer_id, product_id, order_date, quantity) VALUES (?, ?, ?, ?);", [
        (1, 1, "2025-05-01", 1),
        (2, 2, "2025-05-03", 2),
        (3, 3, "2025-05-05", 1),
        (1, 4, "2025-05-10", 3)
    ])

    conn.commit()
    conn.close()
    print(f"Sample database '{db_path}' created successfully.")

if __name__ == "__main__":
    create_sample_db()