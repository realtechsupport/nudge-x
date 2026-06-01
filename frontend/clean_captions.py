#!/usr/bin/env python3
"""
Validate a TSV file and remove quotation marks from the caption/captions column only.

This script is intentionally conservative about "no other changes": it edits the raw
text stream and removes quote characters only while the parser is inside the caption
column. It does not replace embedded line breaks, collapse whitespace, or use
csv.writer, so it will not add structural quotes back into the output.

Usage:
    python3 clean_captions.py input.tsv output.tsv
    python3 clean_captions.py input.tsv output.tsv --expected-columns 6

Important TSV constraint:
    If a caption field contains embedded line breaks, standard CSV/TSV parsers need
    structural quotes to keep that caption as one logical field. Removing those quotes
    while preserving embedded line breaks creates a raw text file with physical line
    breaks inside captions. This script still validates the original logical rows and
    confirms that the cleaned logical rows preserve the original column count.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

DEFAULT_EXPECTED_COLUMNS = 6
CAPTION_COLUMN_NAMES = {"caption", "captions"}
GPS_COLUMN_NAME = "gps_coordinates"
ASCII_QUOTE = '"'
# Include common Unicode quotation marks in case the captions contain typographic quotes.
# ASCII double quote is the main target requested by the user.
QUOTE_CHARS = {
    '"',
    '\u201c', '\u201d', '\u201e', '\u201f',  # curly/low/high double quotes
    '\u00ab', '\u00bb',                       # guillemets
    '\u2039', '\u203a',                       # single guillemets
    '\u301d', '\u301e', '\u301f',             # CJK double quotes
    '\uff02',                                 # fullwidth double quote
}


def normalize_header_name(value: str) -> str:
    return value.strip().lstrip('\ufeff').lower()


def find_column_index(header: Sequence[str], possible_names: Iterable[str]) -> int | None:
    wanted = {name.lower() for name in possible_names}
    for idx, column_name in enumerate(header):
        if normalize_header_name(column_name) in wanted:
            return idx
    return None


def read_tsv_logical_rows(file_path: str | Path) -> List[List[str]]:
    """Read TSV as logical records, respecting quoted embedded newlines."""
    with open(file_path, "r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle, delimiter="\t"))


def validate_rows(rows: Sequence[Sequence[str]], expected_columns: int) -> List[str]:
    issues: List[str] = []

    if not rows:
        return ["File is empty."]

    header = rows[0]
    if len(header) != expected_columns:
        issues.append(f"Header has {len(header)} columns, expected {expected_columns}")

    gps_idx = find_column_index(header, {GPS_COLUMN_NAME})
    if gps_idx is None:
        gps_idx = 4  # Preserve behavior of the original validator.
        issues.append(
            f"Header does not contain '{GPS_COLUMN_NAME}'; using column {gps_idx + 1} for GPS validation"
        )

    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected_columns:
            issues.append(
                f"Row {row_number}: Expected {expected_columns} columns but got {len(row)}"
            )
            continue

        if gps_idx >= len(row):
            issues.append(f"Row {row_number}: Missing {GPS_COLUMN_NAME}")
            continue

        gps = row[gps_idx].strip()
        if not gps or gps.lower() == "unknown":
            issues.append(f"Row {row_number}: Missing or invalid {GPS_COLUMN_NAME}")
            continue

        coords = gps.split(",")
        if len(coords) != 2:
            issues.append(
                f"Row {row_number}: {GPS_COLUMN_NAME} should have 2 values separated by comma"
            )
            continue

        try:
            lat = float(coords[0])
            lon = float(coords[1])
        except ValueError:
            issues.append(f"Row {row_number}: GPS coordinates not valid floats: {gps}")
            continue

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            issues.append(f"Row {row_number}: GPS coordinates out of range: {lat}, {lon}")

    return issues


def strip_line_ending(line: str) -> str:
    return line.rstrip("\r\n")


def split_header_from_text(text: str) -> Tuple[str, str, str]:
    """Return header_without_eol, original_eol, rest_text."""
    for idx, char in enumerate(text):
        if char == "\n":
            if idx > 0 and text[idx - 1] == "\r":
                return text[: idx - 1], "\r\n", text[idx + 1 :]
            return text[:idx], "\n", text[idx + 1 :]
        if char == "\r":
            return text[:idx], "\r", text[idx + 1 :]
    return text, "", ""


def cleaned_logical_rows(rows: Sequence[Sequence[str]], caption_idx: int) -> List[List[str]]:
    cleaned: List[List[str]] = []
    for row_number, row in enumerate(rows):
        new_row = list(row)
        if row_number > 0 and caption_idx < len(new_row):
            new_row[caption_idx] = "".join(ch for ch in new_row[caption_idx] if ch not in QUOTE_CHARS)
        cleaned.append(new_row)
    return cleaned


def assert_column_counts_preserved(
    original_rows: Sequence[Sequence[str]],
    cleaned_rows: Sequence[Sequence[str]],
) -> None:
    if len(original_rows) != len(cleaned_rows):
        raise RuntimeError(
            f"Internal error: logical row count changed from {len(original_rows)} to {len(cleaned_rows)}."
        )
    for row_number, (original_row, cleaned_row) in enumerate(zip(original_rows, cleaned_rows), start=1):
        if len(original_row) != len(cleaned_row):
            raise RuntimeError(
                f"Internal error: logical row {row_number} changed from {len(original_row)} "
                f"columns to {len(cleaned_row)} columns."
            )


def remove_quotes_from_caption_raw_text(text: str, caption_idx: int) -> Tuple[str, int]:
    """Remove quote characters only while scanning the caption/captions column.

    The scanner follows CSV-style quote state so tabs/newlines inside quoted captions
    are treated as caption content, not as delimiters/record endings. Quote characters
    in other columns are left untouched.
    """
    header, eol, rest = split_header_from_text(text)
    output = [header, eol]

    field_idx = 0
    in_quotes = False
    at_field_start = True
    removed = 0
    i = 0

    while i < len(rest):
        ch = rest[i]
        in_caption = field_idx == caption_idx

        if ch == ASCII_QUOTE:
            # Escaped quote inside a quoted field: "" represents one literal quote.
            if in_quotes and i + 1 < len(rest) and rest[i + 1] == ASCII_QUOTE:
                if in_caption:
                    removed += 2
                else:
                    output.append(ch)
                    output.append(rest[i + 1])
                i += 2
                at_field_start = False
                continue

            # Opening/closing quote used as CSV syntax, or a literal quote in an unquoted field.
            if at_field_start and not in_quotes:
                in_quotes = True
            elif in_quotes:
                in_quotes = False
            # If it is a stray quote in the middle of an unquoted field, leave quote state unchanged.

            if in_caption:
                removed += 1
            else:
                output.append(ch)
            at_field_start = False
            i += 1
            continue

        if in_caption and ch in QUOTE_CHARS:
            removed += 1
            at_field_start = False
            i += 1
            continue

        output.append(ch)

        if ch == "\t" and not in_quotes:
            field_idx += 1
            at_field_start = True
        elif ch == "\r" and not in_quotes:
            # Preserve CRLF or bare CR exactly while resetting record state once.
            if i + 1 < len(rest) and rest[i + 1] == "\n":
                output.append(rest[i + 1])
                i += 1
            field_idx = 0
            at_field_start = True
        elif ch == "\n" and not in_quotes:
            field_idx = 0
            at_field_start = True
        else:
            at_field_start = False

        i += 1

    return "".join(output), removed


def count_quote_chars(text: str) -> int:
    return sum(1 for ch in text if ch in QUOTE_CHARS)


def count_ascii_quotes(text: str) -> int:
    return text.count(ASCII_QUOTE)


def clean_file(input_path: str | Path, output_path: str | Path, expected_columns: int) -> int:
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, "r", encoding="utf-8-sig", newline="") as handle:
        original_text = handle.read()
    rows = read_tsv_logical_rows(input_path)

    if not rows:
        print("Input file is empty.", file=sys.stderr)
        return 1

    header = rows[0]
    caption_idx = find_column_index(header, CAPTION_COLUMN_NAMES)
    if caption_idx is None:
        print("Error: Header must contain a 'caption' or 'captions' column.", file=sys.stderr)
        return 1

    input_issues = validate_rows(rows, expected_columns)

    cleaned_text, removed_raw = remove_quotes_from_caption_raw_text(original_text, caption_idx)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(cleaned_text)

    cleaned_rows = cleaned_logical_rows(rows, caption_idx)
    assert_column_counts_preserved(rows, cleaned_rows)
    cleaned_issues = validate_rows(cleaned_rows, expected_columns)

    input_ascii = count_ascii_quotes(original_text)
    output_ascii = count_ascii_quotes(cleaned_text)
    input_all_quotes = count_quote_chars(original_text)
    output_all_quotes = count_quote_chars(cleaned_text)

    print(f"Wrote cleaned TSV: {output_path}")
    print(f"Removed {removed_raw} quote character(s) from the raw caption/captions column text.")
    print(f"Raw ASCII double quote characters in input: {input_ascii}")
    print(f"Raw ASCII double quote characters remaining in output: {output_ascii}")
    if input_all_quotes != input_ascii or output_all_quotes != output_ascii:
        print(f"All supported quote characters in input: {input_all_quotes}")
        print(f"All supported quote characters remaining in output: {output_all_quotes}")

    if input_issues:
        print("\nInput format issues found:")
        for issue in input_issues:
            print(f"- {issue}")
    else:
        print("\nInput TSV format is valid.")

    if cleaned_issues:
        print("\nCleaned logical rows have format issues:")
        for issue in cleaned_issues:
            print(f"- {issue}")
        return 1

    print("\nCleaned logical rows are valid and preserve the original number of columns.")
    return 0 if not input_issues else 1


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a TSV and remove quotation marks from the caption/captions column only."
    )
    parser.add_argument("input_tsv", help="Input TSV path")
    parser.add_argument("output_tsv", help="Output cleaned TSV path")
    parser.add_argument(
        "--expected-columns",
        type=int,
        default=DEFAULT_EXPECTED_COLUMNS,
        help=f"Expected number of columns per logical row. Default: {DEFAULT_EXPECTED_COLUMNS}",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return clean_file(args.input_tsv, args.output_tsv, args.expected_columns)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
