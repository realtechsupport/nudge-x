
import psycopg2
from psycopg2 import Error
import os
from PIL import Image
import io
from typing import List, Tuple

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
                    location VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_accepted BOOLEAN DEFAULT FALSE,
                    is_evaluated BOOLEAN DEFAULT FALSE,
                    caption TEXT
                );
            """)
            # Table for embedding tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS caption_embeddings (
                    id SERIAL PRIMARY KEY,
                    caption_id INT REFERENCES captions(id) ON DELETE CASCADE,
                    embedding_added BOOLEAN DEFAULT FALSE,
                    embedding_created_at TIMESTAMP,
                    qdrant_point_id UUID
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


def save_filename_and_captions(captions_with_metadata: List[Tuple[str, str, str, bool, bool]]):
    """
    Saves multiple image metadata entries and their captions to the PostgreSQL database.
    Returns a list of IDs for the newly saved rows.
    
    Args:
        captions_with_metadata (list of tuples): Each tuple should be 
        (filename, location, caption, is_accepted, is_evaluated)
    """
    conn = connect_db()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()

        insert_sql = """
            INSERT INTO captions (filename, location, caption, is_accepted, is_evaluated)
            VALUES (%s, %s, %s, %s, %s);
        """

        # Execute batch insert
        cursor.executemany(insert_sql, captions_with_metadata)


        conn.commit()
        print(f"Inserted {len(captions_with_metadata)} rows into DB.")
        

    except Exception as e:
        print(f"Error saving captions: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


