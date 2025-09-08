-- Daily aggregation of Reddit comments by subreddit
-- Parameters: @subreddit, @start, @end

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
  AND score >= 0  -- Filter out heavily downvoted comments
GROUP BY
  DATE(TIMESTAMP_SECONDS(created_utc))
ORDER BY
  date ASC