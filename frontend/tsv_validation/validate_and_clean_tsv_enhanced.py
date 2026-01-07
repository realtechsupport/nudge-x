
import csv
import unicodedata

def normalize_text(text):
    return unicodedata.normalize("NFKC", text).strip()

def validate_and_clean_tsv(input_path, output_path):
    issues = []
    expected_columns = 7
    cleaned_rows = []

    with open(input_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='\t')
        raw_header = next(reader, None)
        if raw_header is None:
            issues.append("File is empty.")
            return issues, []

        # Normalize header
        header = [normalize_text(h) for h in raw_header]
        if len(header) != expected_columns:
            issues.append(f"Header has {len(header)} columns, expected {expected_columns}")
        else:
            cleaned_rows.append(header)

        for i, row in enumerate(reader, start=2):
            original_row = row
            row = [normalize_text(col) for col in row]

            if len(row) != expected_columns:
                issues.append(f"Row {i}: Expected {expected_columns} columns but got {len(row)}")
                continue

            # Check for rogue tabs in fields
            if any('\t' in field for field in row):
                issues.append(f"Row {i}: Contains internal tab character in a field.")
                continue

            # Check for newlines inside fields
            if any('\n' in field or '\r' in field for field in row):
                issues.append(f"Row {i}: Contains newline character in a field.")
                continue

            gps = row[5]
            if not gps or gps.lower() == "unknown":
                issues.append(f"Row {i}: Missing or invalid gps_coordinates")
                continue

            coords = gps.split(",")
            if len(coords) != 2:
                issues.append(f"Row {i}: gps_coordinates should have 2 values separated by comma")
                continue

            try:
                lat = float(coords[0])
                lon = float(coords[1])
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    issues.append(f"Row {i}: GPS coordinates out of range: {lat}, {lon}")
                    continue
            except ValueError:
                issues.append(f"Row {i}: GPS coordinates not valid floats: {gps}")
                continue

            cleaned_rows.append(row)

    # Write cleaned file
    with open(output_path, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f, delimiter='\t')
        writer.writerows(cleaned_rows)

    return issues, cleaned_rows


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python validate_and_clean_tsv.py <input.tsv> <output_cleaned.tsv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    result, cleaned = validate_and_clean_tsv(input_file, output_file)

    if result:
        print("Issues found and rows removed:")
        for issue in result:
            print("-", issue)
        print(f"Cleaned file written to: {output_file}")
    else:
        print("No issues found. TSV is valid. Output written to:", output_file)
