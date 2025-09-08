-- Daily aggregation of Reddit posts by subreddit
-- Parameters: @subreddit, @start, @end

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
  AND score >= 0  -- Filter out heavily downvoted posts
GROUP BY
  DATE(TIMESTAMP_SECONDS(created_utc))
ORDER BY
  date ASC