#!/usr/bin/env python3
"""
Collect Reddit posts/comments aggregates from BigQuery's public Reddit dataset.

Outputs per-subreddit daily aggregates filtered by a keyword regex into data/raw/reddit/.

Usage examples:
  python scripts/collect_reddit_bq.py \
    --project nice-script-468809-d6 \
    --start 2021-01-01 --end 2023-12-31 \
    --subreddits wallstreetbets stocks GME amcstock \
    --keyword-pattern '(?i)\b(GME|AMC|BB|KOSS|BBBY|DOGE|SHIB|PEPE|BONK)\b'

Notes:
  - Requires Google Cloud credentials (ADC or service account JSON) and billing enabled.
  - Creates CSV and a small metadata JSON next to it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

from datetime import datetime

from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter, ArrayQueryParameter


DEFAULT_SUBREDDITS = [
    "wallstreetbets", "stocks", "BBBY", "GME", "amcstock",
    "cryptocurrency", "dogecoin", "SHIBArmy", "pepecoin", "bonk",
]

DEFAULT_PATTERN = r"(?i)\b(GME|AMC|BB|KOSS|BBBY|DOGE|SHIB|PEPE|BONK)\b"


POSTS_SQL = """
SELECT
  DATE(TIMESTAMP_SECONDS(created_utc)) AS date,
  COUNT(1) AS post_count,
  SUM(score) AS score_sum,
  AVG(score) AS score_avg,
  SUM(IFNULL(num_comments, 0)) AS comments_sum
FROM `fh-bigquery.reddit_posts`
WHERE
  subreddit IN UNNEST(@subreddits)
  AND DATE(TIMESTAMP_SECONDS(created_utc)) BETWEEN @start AND @end
  AND (
    REGEXP_CONTAINS(LOWER(title), @pattern)
    OR REGEXP_CONTAINS(LOWER(selftext), @pattern)
  )
GROUP BY date
ORDER BY date
"""

COMMENTS_SQL = """
SELECT
  DATE(TIMESTAMP_SECONDS(created_utc)) AS date,
  COUNT(1) AS comment_count,
  SUM(score) AS score_sum,
  AVG(score) AS score_avg
FROM `fh-bigquery.reddit_comments`
WHERE
  subreddit IN UNNEST(@subreddits)
  AND DATE(TIMESTAMP_SECONDS(created_utc)) BETWEEN @start AND @end
  AND REGEXP_CONTAINS(LOWER(body), @pattern)
GROUP BY date
ORDER BY date
"""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_query(client: bigquery.Client, sql: str, subreddits: List[str], start: str, end: str, pattern: str):
    job_config = QueryJobConfig(
        query_parameters=[
            ArrayQueryParameter("subreddits", "STRING", subreddits),
            ScalarQueryParameter("start", "DATE", start),
            ScalarQueryParameter("end", "DATE", end),
            ScalarQueryParameter("pattern", "STRING", pattern.lower()),
        ]
    )
    return client.query(sql, job_config=job_config).result()


def save_result(table, out_csv: Path) -> Dict:
    rows = [dict(row) for row in table]
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    meta = {
        "row_count": len(df),
        "columns": list(df.columns),
    }
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--subreddits", nargs="*", default=DEFAULT_SUBREDDITS)
    parser.add_argument("--keyword-pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--output-dir", default="data/raw/reddit")
    parser.add_argument("--prefix", default="keywords", help="Filename prefix tag")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_dir(Path(args.output_dir))

    # Initialize BQ client
    client = bigquery.Client(project=args.project)

    # Posts
    if args.dry_run:
        print("[DRY RUN] Posts SQL:\n", POSTS_SQL)
        print("[DRY RUN] Comments SQL:\n", COMMENTS_SQL)
        return

    posts_table = run_query(client, POSTS_SQL, args.subreddits, args.start, args.end, args.keyword_pattern)
    comments_table = run_query(client, COMMENTS_SQL, args.subreddits, args.start, args.end, args.keyword_pattern)

    timestamp_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base_name = f"reddit_{args.prefix}_{args.start}_{args.end}".replace("-", "")

    posts_csv = Path(args.output_dir) / f"{base_name}_posts.csv"
    comments_csv = Path(args.output_dir) / f"{base_name}_comments.csv"
    posts_meta_json = Path(args.output_dir) / f"{base_name}_posts.meta.json"
    comments_meta_json = Path(args.output_dir) / f"{base_name}_comments.meta.json"

    posts_meta = save_result(posts_table, posts_csv)
    comments_meta = save_result(comments_table, comments_csv)

    common_meta = {
        "project": args.project,
        "subreddits": args.subreddits,
        "date_range": f"{args.start} to {args.end}",
        "pattern": args.keyword_pattern,
        "collected_at": timestamp_str,
    }

    with open(posts_meta_json, "w") as f:
        json.dump({**common_meta, **posts_meta, "type": "posts"}, f, indent=2)
    with open(comments_meta_json, "w") as f:
        json.dump({**common_meta, **comments_meta, "type": "comments"}, f, indent=2)

    print(f"✅ Saved: {posts_csv} ({posts_meta['row_count']} rows)")
    print(f"✅ Saved: {comments_csv} ({comments_meta['row_count']} rows)")


if __name__ == "__main__":
    main()


