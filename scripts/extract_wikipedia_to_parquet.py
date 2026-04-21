import pandas as pd
import os
import re
from src.utils.wikipedia_extractor import WikipediaExtractor

# Path to the input CSV and output Parquet
CSV_PATH = 'data/raw/RawCombinedDistinctUrls.csv'
PARQUET_PATH = 'data/raw/WikipediaContent_2.parquet'

# Helper to reconstruct split URLs in the CSV
# (Assumes the CSV is not quoted and line breaks split fields)
def read_and_repair_csv(csv_path):
    rows = []
    with open(csv_path, encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        current = []
        for line in f:
            line = line.rstrip('\n')
            # If line looks like a new row (starts with 'http' and has 2 commas), start new row
            if re.match(r'^http[^,]*,[^,]*,https?://', line):
                if current:
                    rows.append(current)
                current = [line]
            else:
                # Continuation of previous row (split URL)
                if current:
                    current[-1] += line
                else:
                    current = [line]
        if current:
            rows.append(current)
    # Split each row into columns
    data = []
    for row in rows:
        joined = ''.join(row)
        parts = joined.split(',')
        if len(parts) >= 3:
            event, eventLabel, wikipediaPage = parts[0], parts[1], ','.join(parts[2:])
            data.append({'event': event, 'eventLabel': eventLabel, 'wikipediaPage': wikipediaPage})
    return pd.DataFrame(data)

def main():
    df = read_and_repair_csv(CSV_PATH)
    extractor = WikipediaExtractor()
    results = []
    for idx, row in df.iterrows():
        url = row['wikipediaPage'].strip()
        event = row['event']
        eventLabel = row['eventLabel']
        try:
            result = extractor.extract_wikipedia_content(url)
            results.append({
                'event': event,
                'eventLabel': eventLabel,
                'url': url,
                'success': result.get('success', False),
                'title': result.get('title', ''),
                'content': result.get('content', ''),
                'error': result.get('error', ''),
                'content_length': result.get('content_length', 0),
            })
        except Exception as e:
            results.append({
                'event': event,
                'eventLabel': eventLabel,
                'url': url,
                'success': False,
                'title': '',
                'content': '',
                'error': str(e),
                'content_length': 0,
            })
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} rows...")
    out_df = pd.DataFrame(results)
    out_df.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved {len(out_df)} rows to {PARQUET_PATH}")

if __name__ == '__main__':
    main() 