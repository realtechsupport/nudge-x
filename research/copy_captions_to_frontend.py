#!/usr/bin/env python3
"""
Script to copy rows from captions table to frontend_captions table.
Joins with METADATA_CSV to get country and GPS coordinates.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from mllm_code.database_pipeline.database_operations import connect_db, create_table_if_not_exists
from mllm_code.config.database_config import POSTGRES_DB

load_dotenv()

def get_metadata_mapping():
    """Load metadata CSV and create a mapping from mine name to metadata."""
    metadata_csv = os.getenv('METADATA_CSV')
    if not metadata_csv:
        raise ValueError("METADATA_CSV environment variable not set")
    
    df = pd.read_csv(metadata_csv)
    
    # Create mapping: mine name -> {country, latitude, longitude}
    # Note: CSV has ' Mine name' with leading space
    mapping = {}
    for _, row in df.iterrows():
        mine_name = str(row[' Mine name']).strip()
        country = str(row['Country']).strip() if pd.notna(row['Country']) else 'Unknown'
        latitude = row['Latitude'] if pd.notna(row['Latitude']) else None
        longitude = row['Longitude'] if pd.notna(row['Longitude']) else None
        
        # Create GPS coordinates string
        if latitude is not None and longitude is not None:
            gps_coords = f"{latitude},{longitude}"
        else:
            gps_coords = "Unknown"
        
        mapping[mine_name] = {
            'country': country,
            'gps_coordinates': gps_coords
        }
    
    return mapping

def copy_captions_to_frontend(only_accepted=True):
    """
    Copy rows from captions to frontend_captions.
    
    Args:
        only_accepted: If True, only copy rows where is_accepted = TRUE
    """
    # Ensure frontend_captions table exists
    print("Ensuring frontend_captions table exists...")
    create_table_if_not_exists()
    
    conn = connect_db()
    if conn is None:
        print("Failed to connect to database")
        return
    
    try:
        # Load metadata mapping
        print("Loading metadata CSV...")
        metadata_mapping = get_metadata_mapping()
        print(f"Loaded metadata for {len(metadata_mapping)} mines")
        
        cursor = conn.cursor()
        
        # Fetch captions to copy
        if only_accepted:
            query = """
                SELECT filename, location, caption
                FROM captions
                WHERE is_accepted = TRUE
                ORDER BY created_at DESC
            """
        else:
            query = """
                SELECT filename, location, caption
                FROM captions
                ORDER BY created_at DESC
            """
        
        cursor.execute(query)
        captions = cursor.fetchall()
        print(f"Found {len(captions)} captions to copy")
        
        # Check if row already exists to avoid duplicates
        check_query = "SELECT COUNT(*) FROM frontend_captions WHERE filename = %s"
        
        # Prepare insert query
        insert_query = """
            INSERT INTO frontend_captions (filename, site_location, country, GPS_coordinates, caption)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        inserted_count = 0
        skipped_count = 0
        missing_metadata_count = 0
        
        for filename, location, caption in captions:
            # Match location with metadata
            location_clean = str(location).strip() if location else None
            
            if not location_clean:
                print(f"Warning: Skipping {filename} - no location")
                skipped_count += 1
                continue
            
            # Try to find metadata - check exact match and partial matches
            metadata = None
            if location_clean in metadata_mapping:
                metadata = metadata_mapping[location_clean]
            else:
                # Try case-insensitive match
                for mine_name, meta in metadata_mapping.items():
                    if mine_name.lower() == location_clean.lower():
                        metadata = meta
                        break
                
                # Try partial match (location contains mine name or vice versa)
                if not metadata:
                    for mine_name, meta in metadata_mapping.items():
                        if location_clean.lower() in mine_name.lower() or mine_name.lower() in location_clean.lower():
                            metadata = meta
                            break
            
            if metadata:
                country = metadata['country']
                gps_coordinates = metadata['gps_coordinates']
            else:
                # Use defaults if metadata not found
                print(f"Warning: No metadata found for location '{location_clean}' (filename: {filename})")
                country = 'Unknown'
                gps_coordinates = 'Unknown'
                missing_metadata_count += 1
            
            # Check if already exists
            cursor.execute(check_query, (filename,))
            exists = cursor.fetchone()[0] > 0
            
            if exists:
                print(f"Skipping {filename} - already exists in frontend_captions")
                skipped_count += 1
                continue
            
            # Insert into frontend_captions
            try:
                cursor.execute(insert_query, (filename, location_clean, country, gps_coordinates, caption))
                inserted_count += 1
            except Exception as e:
                print(f"Error inserting {filename}: {e}")
                skipped_count += 1
        
        conn.commit()
        
        print(f"\n✅ Copy completed!")
        print(f"   Inserted: {inserted_count}")
        print(f"   Skipped: {skipped_count}")
        print(f"   Missing metadata: {missing_metadata_count}")
        
        # Verify the copy
        cursor.execute("SELECT COUNT(*) FROM frontend_captions")
        total_count = cursor.fetchone()[0]
        print(f"   Total rows in frontend_captions: {total_count}")
        
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
        raise
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Copy captions to frontend_captions table")
    parser.add_argument("--all", action="store_true", help="Copy all captions (not just accepted ones)")
    
    args = parser.parse_args()
    
    copy_captions_to_frontend(only_accepted=not args.all)

