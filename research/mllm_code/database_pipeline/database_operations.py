
import psycopg2
from psycopg2 import Error
import os
from PIL import Image
import io
from typing import List, Tuple

from mllm_code.config.database_config import POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT

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
        raise

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
                    embedding_created_at TIMESTAMP
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




def fetch_captions_without_embeddings(limit: int = 50):
    """
    Returns a list of captions that are accepted and do not yet have embeddings added.

    Each row is returned as a dict with keys matching expected usage:
    id, filename, location, caption, is_accepted, is_evaluated, created_at
    """
    conn = connect_db()
    if conn is None:
        return []

    try:
        cursor = conn.cursor()
        # Select captions that are accepted AND either have no row in caption_embeddings
        # or have a row with embedding_added = FALSE
        query = """
            SELECT c.id, c.filename, c.location, c.caption, c.is_accepted, c.is_evaluated, c.created_at
            FROM captions c
            LEFT JOIN caption_embeddings ce ON ce.caption_id = c.id
            WHERE c.is_accepted = TRUE
              AND (ce.id IS NULL OR ce.embedding_added = FALSE)
            ORDER BY c.created_at DESC
            LIMIT %s;
        """
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        result = [dict(zip(columns, row)) for row in rows]
        return result
    except Exception as e:
        print(f"Error fetching captions without embeddings: {e}")
        return []
    finally:
        if conn:
            cursor.close()
            conn.close()


def mark_embeddings_added(caption_ids: List[int]):
    """
    Marks embeddings as added for the provided caption IDs.

    Args:
        caption_ids: list of caption_id values to mark as embedded
    """
    if not caption_ids:
        return

    conn = connect_db()
    if conn is None:
        return

    try:
        cursor = conn.cursor()

        # For each caption_id, try UPDATE first; if no row was updated, INSERT a new one
        update_sql = """
            UPDATE caption_embeddings
               SET embedding_added = TRUE,
                   embedding_created_at = CURRENT_TIMESTAMP
             WHERE caption_id = %s;
        """
        insert_sql = """
            INSERT INTO caption_embeddings (caption_id, embedding_added, embedding_created_at)
            VALUES (%s, TRUE, CURRENT_TIMESTAMP);
        """

        for caption_id in caption_ids:
            cursor.execute(update_sql, (caption_id,))
            if cursor.rowcount == 0:
                cursor.execute(insert_sql, (caption_id,))

        conn.commit()
    except Exception as e:
        print(f"Error marking embeddings added: {e}")
        conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()
