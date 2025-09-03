# Reddit Collector Usage Guide

## Prerequisites

1. Install required packages:
```bash
pip install google-cloud-bigquery pandas python-dateutil
```

2. Set up BigQuery authentication:
```bash
# Option 1: Service account key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"

# Option 2: Application default credentials
gcloud auth application-default login
```

## Basic Usage

### Collect Standard Reddit Data

```bash
python collect_reddit.py \
  --subreddits wallstreetbets stocks GME amcstock \
  --start 2021-01-01 --end 2021-12-31 \
  --project your-gcp-project-id
```

### Collect with Keyword Filtering

```bash
python collect_reddit.py \
  --subreddits wallstreetbets stocks BBBY GME amcstock cryptocurrency dogecoin \
  --start 2020-12-01 --end 2023-12-31 \
  --use-keywords true \
  --keyword-pattern '(?i)\b(GME|AMC|BB|KOSS|BBBY|DOGE|SHIB|PEPE|BONK)\b' \
  --project your-gcp-project-id
```

### Dry Run (Test Without Writing Files)

```bash
python collect_reddit.py \
  --subreddits wallstreetbets \
  --start 2021-01-01 --end 2021-01-07 \
  --project your-gcp-project-id \
  --dry-run true
```

## Output Files

For each subreddit, the collector generates:

### CSV Files
- `data/raw/reddit/reddit_{subreddit}.csv` - Base data
- `data/raw/reddit/reddit_{subreddit}_keywords.csv` - Keyword-filtered data (if enabled)

### Schema (Required Columns)
```csv
date,posts,comments,score,total_engagement,is_weekend
2021-01-01,45,128,573,846,0
2021-01-02,12,67,234,313,1
```

### Metadata Files
- `data/raw/reddit/reddit_{subreddit}.meta.json` - Metadata with collection info
- Automatically appended to `data/INDEX.jsonl` for tracking

## SQL Templates

The collector uses SQL templates from the `sql/` directory:

- `posts_daily.sql` - Daily post aggregation
- `comments_daily.sql` - Daily comment aggregation  
- `posts_daily_keywords.sql` - Post aggregation with keyword filtering
- `comments_daily_keywords.sql` - Comment aggregation with keyword filtering

### Template Parameters
- `@subreddit` - Subreddit name
- `@start` - Start date (YYYY-MM-DD)
- `@end` - End date (YYYY-MM-DD)
- `@keyword_pattern` - Regex pattern for keyword filtering (keywords templates only)

## Field Calculations

The collector computes the following fields:

- **posts**: Count of posts per day
- **comments**: Count of comments per day  
- **score**: Sum of post scores + comment scores
- **total_engagement**: Sum of post scores + comment counts + comment scores
- **is_weekend**: 1 if Saturday/Sunday, 0 otherwise

## Error Handling

- **Authentication errors**: Clear message about setting up credentials
- **Rate limiting**: Automatic retry with exponential backoff
- **Data validation**: Ensures no duplicates, ascending dates, non-negative counts
- **Missing templates**: Helpful error message with expected file path

## Logging

Structured logs are written to `logs/{YYYYMMDD}.log` with the following format:

```
[RUN] subreddit=wallstreetbets start=2021-01-01 end=2021-12-31 use_keywords=false
[BQ] posts_rows=365 comments_rows=365
[JOIN] out_rows=365 missing_dates=0
[VALIDATION] symbol=r/wallstreetbets no_duplicates=True monotonic_dates=True non_negative_counts=True
[WRITE] path=data/raw/reddit/reddit_wallstreetbets.csv rows=365 checksum=a1b2c3d4... version=20250812153000
```

## Troubleshooting

### "BigQuery authentication failed"
- Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
- Or run `gcloud auth application-default login`

### "SQL template not found"
- Ensure `sql/` directory exists with required `.sql` files
- Check template names match expected patterns

### "No data returned"
- Verify subreddit names are correct (without r/ prefix)
- Check date range is valid for the data available in BigQuery
- Ensure BigQuery dataset access permissions

## Mock Testing

For development/testing without BigQuery:

```bash
python test_reddit_collector.py  # Test individual components
python test_reddit_mock.py       # Test full pipeline with fake data
```