#!/usr/bin/env python3
"""
Script to copy rows from captions table to frontend_captions table.
Uses METADATA_CSV only to get GPS coordinates (matched by mine_name).
All other fields (filename, mine_name, site_location, country, caption) are copied directly from captions table.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from mllm_code.database_pipeline.database_operations import connect_db, create_table_if_not_exists
from mllm_code.config.database_config import POSTGRES_DB

load_dotenv()

def get_gps_mapping():
    """Load metadata CSV and create a mapping from mine name to GPS coordinates."""
    metadata_csv = os.getenv('METADATA_CSV')
    if not metadata_csv:
        raise ValueError("METADATA_CSV environment variable not set")
    
    df = pd.read_csv(metadata_csv)
    
    # Create mapping: mine name -> gps_coordinates
    mapping = {}
    for _, row in df.iterrows():
        mine_name = str(row['Mine name']).strip()
        
        # Try to get coordinates - handle multiple data entry formats
        latitude = None
        longitude = None
        
        # First, try the separate Latitude/Longitude columns
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            latitude = row['Latitude']
            longitude = row['Longitude']
        # If not available, try to parse the combined Lat/Long column
        elif pd.notna(row.get('Lat/Long')) and str(row['Lat/Long']).strip():
            lat_long_str = str(row['Lat/Long']).strip()
            # Remove any surrounding quotes
            lat_long_str = lat_long_str.strip('"\'')
            # Parse combined format like "-32.443459, 151.014824" or "-32.443459,151.014824"
            if ',' in lat_long_str:
                parts = lat_long_str.split(',')
                if len(parts) == 2:
                    try:
                        latitude = float(parts[0].strip())
                        longitude = float(parts[1].strip())
                    except ValueError:
                        # If parsing fails, coordinates remain None
                        pass
        
        # Create GPS coordinates string
        if latitude is not None and longitude is not None:
            gps_coords = f"{latitude},{longitude}"
        else:
            gps_coords = "Unknown"
        
        mapping[mine_name] = gps_coords
    
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
        # Load GPS coordinates mapping from metadata CSV
        print("Loading metadata CSV for GPS coordinates...")
        gps_mapping = get_gps_mapping()
        print(f"Loaded GPS coordinates for {len(gps_mapping)} mines")
        
        cursor = conn.cursor()
        
        # Fetch captions to copy (all fields from captions table)
        if only_accepted:
            query = """
                SELECT filename, mine_name, location, country, caption
                FROM captions
                WHERE is_accepted = TRUE
                ORDER BY created_at DESC
            """
        else:
            query = """
                SELECT filename, mine_name, location, country, caption
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
            INSERT INTO frontend_captions (filename, mine_name, site_location, country, GPS_coordinates, caption)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        inserted_count = 0
        skipped_count = 0
        missing_gps_count = 0
        
        for filename, mine_name, location, country, caption in captions:
            mine_name_clean = str(mine_name).strip() if mine_name else 'Unknown'
            location_clean = str(location).strip() if location else 'Unknown'
            country_clean = str(country).strip() if country else 'Unknown'
            
            # Look up GPS coordinates using mine_name (exact match, then case-insensitive)
            gps_coordinates = None
            if mine_name_clean in gps_mapping:
                gps_coordinates = gps_mapping[mine_name_clean]
            else:
                # Try case-insensitive match 
                for csv_mine_name, gps in gps_mapping.items():
                    if csv_mine_name.lower() == mine_name_clean.lower():
                        gps_coordinates = gps
                        break
            
            if gps_coordinates is None:
                print(f"Warning: No GPS coordinates found for mine '{mine_name_clean}' (filename: {filename})")
                gps_coordinates = 'Unknown'
                missing_gps_count += 1 
            
            # Check if already exists
            cursor.execute(check_query, (filename,))
            exists = cursor.fetchone()[0] > 0
            
            if exists:
                print(f"Skipping {filename} - already exists in frontend_captions")
                skipped_count += 1
                continue
            
            # Insert into frontend_captions
            try:
                cursor.execute(insert_query, (filename, mine_name_clean, location_clean, country_clean, gps_coordinates, caption))
                inserted_count += 1
            except Exception as e:
                print(f"Error inserting {filename}: {e}")
                skipped_count += 1
        
        conn.commit()
        
        print(f"\n✅ Copy completed!")
        print(f"   Inserted: {inserted_count}")
        print(f"   Skipped: {skipped_count}")
        print(f"   Missing GPS coordinates: {missing_gps_count}")
        
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

