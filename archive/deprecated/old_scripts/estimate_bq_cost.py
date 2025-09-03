#!/usr/bin/env python3
"""
Estimate BigQuery cost (USD) via dry-run for Reddit posts/comments by year.

Usage:
  export GCP_PROJECT="nice-script-468809-d6"
  python scripts/estimate_bq_cost.py \
    --years 2021 2022 2023 \
    --subreddits wallstreetbets stocks BBBY GME amcstock cryptocurrency dogecoin SHIBArmy pepecoin bonk \
    --pattern '(?i)\\b(GME|AMC|BB|KOSS|BBBY|DOGE|SHIB|PEPE|BONK)\\b'

Notes:
  - Requires google-cloud-bigquery and credentials (ADC or service account).
  - Pricing assumed at $5 per TB scanned.
"""

from __future__ import annotations

import argparse
from typing import List

from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter, ArrayQueryParameter

PRICE_PER_TB = 5.0

POSTS_SQL = """
SELECT DATE(TIMESTAMP_SECONDS(created_utc)) d
FROM `fh-bigquery.reddit_posts`
WHERE subreddit IN UNNEST(@subs)
  AND DATE(TIMESTAMP_SECONDS(created_utc)) BETWEEN @start AND @end
  AND (
    REGEXP_CONTAINS(LOWER(title), @pattern)
    OR REGEXP_CONTAINS(LOWER(selftext), @pattern)
  )
"""

COMMENTS_SQL = """
SELECT DATE(TIMESTAMP_SECONDS(created_utc)) d
FROM `fh-bigquery.reddit_comments`
WHERE subreddit IN UNNEST(@subs)
  AND DATE(TIMESTAMP_SECONDS(created_utc)) BETWEEN @start AND @end
  AND REGEXP_CONTAINS(LOWER(body), @pattern)
"""


def tb_to_usd(total_bytes: int) -> float:
    return (total_bytes / (1024 ** 4)) * PRICE_PER_TB


def gb(total_bytes: int) -> float:
    return total_bytes / (1024 ** 3)


def dry_run_bytes(client: bigquery.Client, sql: str, subs: List[str], start: str, end: str, pattern: str) -> int:
    job_config = QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
        query_parameters=[
            ArrayQueryParameter("subs", "STRING", subs),
            ScalarQueryParameter("start", "DATE", start),
            ScalarQueryParameter("end", "DATE", end),
            ScalarQueryParameter("pattern", "STRING", pattern.lower()),
        ],
    )
    job = client.query(sql, job_config=job_config)
    return int(job.total_bytes_processed or 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", required=True)
    parser.add_argument("--subreddits", nargs="+", required=True)
    parser.add_argument("--pattern", default=r"(?i)\b(GME|AMC|BB|KOSS|BBBY|DOGE|SHIB|PEPE|BONK)\b")
    parser.add_argument("--project", default=None, help="GCP project (optional if set via env)")
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)

    grand_total_bytes = 0
    print("Pricing: $%.2f per TB scanned\n" % PRICE_PER_TB)
    for y in args.years:
        start = f"{y}-01-01"
        end = f"{y}-12-31"

        posts_bytes = dry_run_bytes(client, POSTS_SQL, args.subreddits, start, end, args.pattern)
        comments_bytes = dry_run_bytes(client, COMMENTS_SQL, args.subreddits, start, end, args.pattern)

        year_total = posts_bytes + comments_bytes
        grand_total_bytes += year_total

        print(f"{y} posts:    {gb(posts_bytes):8.3f} GB  ~ $ {tb_to_usd(posts_bytes):.2f}")
        print(f"{y} comments: {gb(comments_bytes):8.3f} GB  ~ $ {tb_to_usd(comments_bytes):.2f}")
        print(f"{y} total:    {gb(year_total):8.3f} GB  ~ $ {tb_to_usd(year_total):.2f}")
        print("-")

    print("Grand total:")
    print(f"  {gb(grand_total_bytes):.3f} GB  ~ $ {tb_to_usd(grand_total_bytes):.2f}")


if __name__ == "__main__":
    main()


