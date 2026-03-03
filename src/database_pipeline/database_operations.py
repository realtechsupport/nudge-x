import psycopg2
from psycopg2 import Error
import os
from typing import List, Tuple, Optional

from mllm.config.database_config import POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT

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
            
            # Table for pipeline run metadata - MUST be created and committed first
            # because other tables reference it via foreign key
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS caption_pipeline_runs (
                    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    prompt_version VARCHAR(100),
                    model_name VARCHAR(255),
                    temperature FLOAT,
                    frequency_penalty FLOAT,
                    top_p FLOAT,
                    num_shots INT
                );
            """)
            # Commit to ensure caption_pipeline_runs exists before FK references
            conn.commit()
            
            # Table for images and captions - create BEFORE migrations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS captions (
                    id SERIAL PRIMARY KEY,
                    run_id UUID REFERENCES caption_pipeline_runs(run_id) ON DELETE SET NULL,
                    filename VARCHAR(255) NOT NULL,
                    mine_name VARCHAR(255),
                    location VARCHAR(255),
                    country VARCHAR(255),
                    latitude FLOAT,
                    longitude FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_accepted BOOLEAN DEFAULT FALSE,
                    is_evaluated BOOLEAN DEFAULT FALSE,
                    question TEXT,
                    caption TEXT
                );
            """)
            conn.commit()
            
            # Add new columns to existing captions table if they don't exist (migration)
            # This only runs if table already existed without these columns
            cursor.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name='captions' AND column_name='run_id') THEN
                        ALTER TABLE captions ADD COLUMN run_id UUID REFERENCES caption_pipeline_runs(run_id) ON DELETE SET NULL;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name='captions' AND column_name='latitude') THEN
                        ALTER TABLE captions ADD COLUMN latitude FLOAT;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name='captions' AND column_name='longitude') THEN
                        ALTER TABLE captions ADD COLUMN longitude FLOAT;
                    END IF;
                END $$;
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

            # Create view for frontend captions (replaces frontend_captions table)
            cursor.execute("""
                CREATE OR REPLACE VIEW frontend_captions_view AS
                SELECT 
                    c.id,
                    c.filename,
                    c.mine_name,
                    c.location AS site_location,
                    c.country,
                    CASE 
                        WHEN c.latitude IS NOT NULL AND c.longitude IS NOT NULL 
                        THEN c.latitude || ',' || c.longitude 
                        ELSE 'Unknown' 
                    END AS GPS_coordinates,
                    c.caption
                FROM captions c
                WHERE c.is_accepted = TRUE
                ORDER BY c.created_at DESC;
            """)
            
            conn.commit()
            print("Tables and views checked/created successfully.")
        except Error as e:
            print(f"Error creating tables: {e}")
        finally:
            if conn:
                cursor.close()
                conn.close()


def create_pipeline_run(
    prompt_version: str,
    model_name: str,
    temperature: float,
    frequency_penalty: float,
    top_p: float,
    num_shots: int
) -> Optional[str]:
    """
    Creates a new pipeline run record and returns the run_id (UUID).
    
    Args:
        prompt_version: Version identifier for the prompt used
        model_name: Name of the model used for generation
        temperature: Temperature setting for generation
        frequency_penalty: Frequency penalty setting
        top_p: Top-p (nucleus sampling) setting
        num_shots: Number of few-shot examples used
    
    Returns:
        run_id (UUID string) if successful, None otherwise
    """
    # Ensure tables exist before inserting
    create_table_if_not_exists()
    
    conn = connect_db()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()
        
        insert_sql = """
            INSERT INTO caption_pipeline_runs 
                (prompt_version, model_name, temperature, frequency_penalty, top_p, num_shots)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING run_id;
        """
        
        cursor.execute(insert_sql, (
            prompt_version, model_name, temperature, frequency_penalty,
            top_p, num_shots
        ))
        
        run_id = cursor.fetchone()[0]
        conn.commit()
        print(f"Created pipeline run with ID: {run_id}")
        return str(run_id)
        
    except Exception as e:
        print(f"Error creating pipeline run: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()


def save_filename_and_captions(
    captions_with_metadata: List[Tuple[str, str, str, str, str, bool, bool, str, Optional[float], Optional[float]]],
    run_id: Optional[str] = None
):
    """
    Saves multiple image metadata entries and their captions to the PostgreSQL database.
    
    Args:
        captions_with_metadata (list of tuples): Each tuple should be 
            (filename, mine_name, location, country, caption, is_accepted, is_evaluated, question, latitude, longitude)
        run_id: Optional UUID string linking to caption_pipeline_runs table. 
                If provided, must exist in caption_pipeline_runs table.
    """
    # Ensure tables exist before inserting
    create_table_if_not_exists()
    
    conn = connect_db()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()
        
        # Validate run_id exists if provided
        if run_id is not None:
            cursor.execute(
                "SELECT 1 FROM caption_pipeline_runs WHERE run_id = %s;",
                (run_id,)
            )
            if cursor.fetchone() is None:
                print(f"Error: run_id '{run_id}' does not exist in caption_pipeline_runs table.")
                print("Please create a pipeline run first using create_pipeline_run().")
                cursor.close()
                conn.close()
                return None

        insert_sql = """
            INSERT INTO captions (run_id, filename, mine_name, location, country, caption, is_accepted, is_evaluated, question, latitude, longitude)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """

        # Prepend run_id to each tuple
        rows_to_insert = [
            (run_id, *row) for row in captions_with_metadata
        ]
        
        # Execute batch insert
        cursor.executemany(insert_sql, rows_to_insert)

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
    id, filename, mine_name, location, country, latitude, longitude, caption, is_accepted, is_evaluated, created_at
    """
    conn = connect_db()
    if conn is None:
        return []

    try:
        cursor = conn.cursor()
        # Select captions that are accepted AND either have no row in caption_embeddings
        # or have a row with embedding_added = FALSE
        # Now includes mine_name, country, latitude, longitude for metadata enrichment
        query = """
            SELECT c.id, c.filename, c.mine_name, c.location, c.country, 
                   c.latitude, c.longitude, c.caption, c.is_accepted, c.is_evaluated, c.created_at
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


def fetch_stale_embedding_caption_ids(limit: int = 500) -> List[int]:
    """
    Return caption IDs whose embeddings are considered stale.

    A caption is treated as stale if:
      - it is accepted AND
      - embedding_added = TRUE in caption_embeddings AND
      - there exists a *newer* accepted caption with the same filename.

    These represent older embeddings that should be deleted from Qdrant so that
    only embeddings for the latest accepted captions per image remain.
    """
    conn = connect_db()
    if conn is None:
        return []

    try:
        cursor = conn.cursor()
        query = """
            SELECT ce.caption_id
            FROM caption_embeddings ce
            JOIN captions c ON ce.caption_id = c.id
            WHERE c.is_accepted = TRUE
              AND ce.embedding_added = TRUE
              AND EXISTS (
                  SELECT 1
                  FROM captions c2
                  WHERE c2.filename = c.filename
                    AND c2.is_accepted = TRUE
                    AND c2.created_at > c.created_at
              )
            ORDER BY c.created_at DESC
            LIMIT %s;
        """
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        print(f"Error fetching stale embedding caption IDs: {e}")
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
