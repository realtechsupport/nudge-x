
import csv

def validate_tsv(file_path):
    issues = []
    expected_columns = 7

    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        if header is None:
            issues.append("File is empty.")
            return issues

        if len(header) != expected_columns:
            issues.append(f"Header has {len(header)} columns, expected {expected_columns}")

        for i, row in enumerate(reader, start=2):  # start=2 to account for header row
            if len(row) != expected_columns:
                issues.append(f"Row {i}: Expected {expected_columns} columns but got {len(row)}")
                continue

            gps = row[5].strip()
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
            except ValueError:
                issues.append(f"Row {i}: GPS coordinates not valid floats: {gps}")

    return issues


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python validate_tsv.py <file.tsv>")
        sys.exit(1)

    file_path = sys.argv[1]
    result = validate_tsv(file_path)

    if result:
        print("Issues found:")
        for issue in result:
            print("-", issue)
    else:
        print("No issues found. TSV is valid.")
