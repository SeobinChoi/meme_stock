-- Daily posts aggregate filtered by keyword regex and subreddit list
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
ORDER BY date;

-- Daily aggregation of Reddit posts by subreddit with keyword filtering
-- Parameters: @subreddit, @start, @end, @keyword_pattern

SELECT
  DATE(TIMESTAMP_SECONDS(created_utc)) as date,
  COUNT(*) as posts,
  SUM(score) as score_sum,
  SUM(num_comments) as comments_sum
FROM 
  `fh-bigquery.reddit_posts.2020_*`
WHERE
  subreddit = @subreddit
  AND DATE(TIMESTAMP_SECONDS(created_utc)) >= @start
  AND DATE(TIMESTAMP_SECONDS(created_utc)) <= @end
  AND score >= 0
  AND (
    REGEXP_CONTAINS(title, @keyword_pattern)
    OR REGEXP_CONTAINS(selftext, @keyword_pattern)
  )
GROUP BY
  DATE(TIMESTAMP_SECONDS(created_utc))
ORDER BY
  date ASC