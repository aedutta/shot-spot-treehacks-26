#!/usr/bin/env python3
"""
URL Extractor for Video Dataset
Extracts URLs from the second column of a CSV dataset for semantic embedding pipeline.

Usage:
    python extract_urls.py <path_to_csv>

Example:
    python extract_urls.py warriors_highlights.csv
"""

import sys
import csv
import json
from pathlib import Path
from typing import List


def extract_urls_from_dataset_csv(csv_path: str, column_index: int = 1) -> List[str]:
    """
    Extract URLs from a specific column of a CSV file.

    Args:
        csv_path: Path to the CSV file
        column_index: Index of the column containing URLs (default: 1 for second column)

    Returns:
        List of URLs from the specified column

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV is empty or column index is invalid
    """
    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    urls = []

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        try:
            header = next(reader)
            if column_index >= len(header):
                raise ValueError(
                    f"Column index {column_index} out of range. "
                    f"CSV has {len(header)} columns."
                )
        except StopIteration:
            raise ValueError("CSV file is empty")

        for row_num, row in enumerate(reader, start=2):
            if len(row) > column_index:
                url = row[column_index].strip()
                if url:
                    urls.append(url)
            else:
                print(f"Warning: Row {row_num} has fewer columns than expected", 
                      file=sys.stderr)

    return urls


def extract_urls_by_name(csv_path: str, column_name: str = "url") -> List[str]:
    """
    Extract URLs from a named column of a CSV file.

    Args:
        csv_path: Path to the CSV file
        column_name: Name of the column containing URLs (default: "url")

    Returns:
        List of URLs from the specified column

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the column name is not found
    """
    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    urls = []

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        if column_name not in reader.fieldnames:
            raise ValueError(
                f"Column '{column_name}' not found. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            url = row[column_name].strip()
            if url:
                urls.append(url)

    return urls


def save_urls(urls: List[str], output_path: str, format: str = 'json') -> None:
    """
    Save URLs to a file in specified format.

    Args:
        urls: List of URLs to save
        output_path: Path for the output file
        format: Output format - 'json', 'txt', or 'csv' (default: 'json')
    """
    output_file = Path(output_path)

    if format == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"urls": urls}, f, indent=2)
    elif format == 'txt':
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(urls))
    elif format == 'csv':
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['url'])
            for url in urls:
                writer.writerow([url])
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Saved {len(urls)} URLs to {output_path}")


def main():
    """Main function for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: python extract_urls.py <path_to_csv> [output_path] [format]")
        print("\nFormats: json (default), txt, csv")
        print("\nExample:")
        print("  python extract_urls.py warriors_highlights.csv urls.json json")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_format = sys.argv[3] if len(sys.argv) > 3 else 'json'

    try:
        # Extract URLs (using second column by default)
        urls = extract_urls(csv_path, column_index=1)

        print(f"Extracted {len(urls)} URLs from {csv_path}")

        # Save or print URLs
        if output_path:
            save_urls(urls, output_path, output_format)
        else:
            # Print to stdout as JSON for piping to other scripts
            print(json.dumps({"urls": urls}, indent=2))

        return urls

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
