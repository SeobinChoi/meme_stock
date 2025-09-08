-- Daily comments aggregate filtered by keyword regex and subreddit list
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
ORDER BY date;

-- Daily aggregation of Reddit comments by subreddit with keyword filtering
-- Parameters: @subreddit, @start, @end, @keyword_pattern

SELECT
  DATE(TIMESTAMP_SECONDS(created_utc)) as date,
  COUNT(*) as comments,
  SUM(score) as comment_score_sum
FROM 
  `fh-bigquery.reddit_comments.2020_*`
WHERE
  subreddit = @subreddit
  AND DATE(TIMESTAMP_SECONDS(created_utc)) >= @start
  AND DATE(TIMESTAMP_SECONDS(created_utc)) <= @end
  AND score >= 0
  AND REGEXP_CONTAINS(body, @keyword_pattern)
GROUP BY
  DATE(TIMESTAMP_SECONDS(created_utc))
ORDER BY
  date ASC