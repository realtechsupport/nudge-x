
import psycopg2
from psycopg2 import Error
import os
from PIL import Image
import io

from config.database_config import POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT

def connect_db():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        return conn
    except Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def create_table_if_not_exists():
    """Creates necessary tables in the database if they don't exist."""
    conn = connect_db()
    if conn:
        try:
            cursor = conn.cursor()
            # Table for images and captions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS captions (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_accepted BOOLEAN,
                    captions TEXT[]
                );
            """)
            conn.commit()
            print("Tables checked/created successfully.")
        except Error as e:
            print(f"Error creating tables: {e}")
        finally:
            if conn:
                cursor.close()
                conn.close()

def save_image_and_captions(image_path, captions, is_accepted):
    """
    Saves the image metadata and its captions to the PostgreSQL database.
    Returns the ID of the newly saved row.
    """
    conn = connect_db()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()
        filename = os.path.basename(image_path)

        insert_sql = """
            INSERT INTO captions (filename, is_accepted, captions)
            VALUES (%s, %s, %s) RETURNING id;
        """
        cursor.execute(insert_sql, (filename, is_accepted, captions))
        image_id = cursor.fetchone()[0]
        conn.commit()
        print(f"Caption(s) for '{filename}' saved to DB with ID: {image_id}")
        return image_id
    except Error as e:
        print(f"Error saving caption(s): {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()

