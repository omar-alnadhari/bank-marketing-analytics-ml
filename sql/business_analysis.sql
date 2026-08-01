-- name: dataset_overview
-- High-level dataset and conversion summary.

SELECT
    COUNT(*) AS total_clients,
    SUM(subscribed) AS subscribers,
    COUNT(*) - SUM(subscribed) AS non_subscribers,
    ROUND(
        100.0 * SUM(subscribed) / COUNT(*),
        2
    ) AS overall_subscription_rate
FROM bank_marketing;


-- name: subscription_by_job
-- Customer volume and subscription performance by job.

SELECT
    job,
    COUNT(*) AS client_count,
    SUM(subscribed) AS subscriber_count,
    ROUND(
        100.0 * SUM(subscribed) / COUNT(*),
        2
    ) AS subscription_rate,
    ROUND(AVG(balance), 2) AS average_balance
FROM bank_marketing
GROUP BY job
ORDER BY
    subscription_rate DESC,
    client_count DESC;


-- name: subscription_by_education
-- Subscription performance by education level.

SELECT
    education,
    COUNT(*) AS client_count,
    SUM(subscribed) AS subscriber_count,
    ROUND(
        100.0 * SUM(subscribed) / COUNT(*),
        2
    ) AS subscription_rate,
    ROUND(AVG(age), 2) AS average_age,
    ROUND(AVG(balance), 2) AS average_balance
FROM bank_marketing
GROUP BY education
ORDER BY
    subscription_rate DESC,
    client_count DESC;


-- name: subscription_by_contact_type
-- Subscription performance by communication channel.

SELECT
    contact,
    COUNT(*) AS client_count,
    SUM(subscribed) AS subscriber_count,
    ROUND(
        100.0 * SUM(subscribed) / COUNT(*),
        2
    ) AS subscription_rate,
    ROUND(
        AVG(call_duration_seconds),
        2
    ) AS average_call_duration_seconds
FROM bank_marketing
GROUP BY contact
ORDER BY
    subscription_rate DESC,
    client_count DESC;


-- name: subscription_by_month
-- Monthly campaign performance using a CTE for calendar ordering.

WITH month_lookup (
    month_name,
    month_number
) AS (
    VALUES
        ('jan', 1),
        ('feb', 2),
        ('mar', 3),
        ('apr', 4),
        ('may', 5),
        ('jun', 6),
        ('jul', 7),
        ('aug', 8),
        ('sep', 9),
        ('oct', 10),
        ('nov', 11),
        ('dec', 12)
)
SELECT
    marketing.contact_month,
    month_lookup.month_number,
    COUNT(*) AS client_count,
    SUM(marketing.subscribed) AS subscriber_count,
    ROUND(
        100.0
        * SUM(marketing.subscribed)
        / COUNT(*),
        2
    ) AS subscription_rate
FROM bank_marketing AS marketing
INNER JOIN month_lookup
    ON marketing.contact_month
       = month_lookup.month_name
GROUP BY
    marketing.contact_month,
    month_lookup.month_number
ORDER BY month_lookup.month_number;


-- name: subscription_by_previous_outcome
-- Performance by previous campaign outcome.

SELECT
    previous_outcome,
    COUNT(*) AS client_count,
    SUM(subscribed) AS subscriber_count,
    ROUND(
        100.0 * SUM(subscribed) / COUNT(*),
        2
    ) AS subscription_rate,
    ROUND(AVG(previous_contacts), 2)
        AS average_previous_contacts
FROM bank_marketing
GROUP BY previous_outcome
ORDER BY
    subscription_rate DESC,
    client_count DESC;


-- name: average_balance_by_subscription
-- Account-balance comparison between subscribers and non-subscribers.

SELECT
    subscribed,
    CASE
        WHEN subscribed = 1
            THEN 'Subscription'
        ELSE 'No subscription'
    END AS subscription_label,
    COUNT(*) AS client_count,
    ROUND(AVG(balance), 2) AS average_balance,
    MIN(balance) AS minimum_balance,
    MAX(balance) AS maximum_balance
FROM bank_marketing
GROUP BY subscribed
ORDER BY subscribed;


-- name: conversion_by_age_group
-- Conversion rates across business-friendly age groups.

WITH age_segments AS (
    SELECT
        CASE
            WHEN age BETWEEN 18 AND 29
                THEN '18-29'
            WHEN age BETWEEN 30 AND 39
                THEN '30-39'
            WHEN age BETWEEN 40 AND 49
                THEN '40-49'
            WHEN age BETWEEN 50 AND 59
                THEN '50-59'
            WHEN age BETWEEN 60 AND 69
                THEN '60-69'
            ELSE '70+'
        END AS age_group,
        subscribed
    FROM bank_marketing
)
SELECT
    age_group,
    COUNT(*) AS client_count,
    SUM(subscribed) AS subscriber_count,
    ROUND(
        100.0 * SUM(subscribed) / COUNT(*),
        2
    ) AS subscription_rate
FROM age_segments
GROUP BY age_group
ORDER BY
    CASE age_group
        WHEN '18-29' THEN 1
        WHEN '30-39' THEN 2
        WHEN '40-49' THEN 3
        WHEN '50-59' THEN 4
        WHEN '60-69' THEN 5
        WHEN '70+' THEN 6
    END;


-- name: conversion_by_campaign_contacts
-- Conversion rates by current-campaign contact intensity.

WITH campaign_segments AS (
    SELECT
        CASE
            WHEN campaign_contacts = 1
                THEN '1 contact'
            WHEN campaign_contacts BETWEEN 2 AND 3
                THEN '2-3 contacts'
            WHEN campaign_contacts BETWEEN 4 AND 5
                THEN '4-5 contacts'
            WHEN campaign_contacts BETWEEN 6 AND 10
                THEN '6-10 contacts'
            ELSE '11+ contacts'
        END AS campaign_contact_group,
        subscribed
    FROM bank_marketing
)
SELECT
    campaign_contact_group,
    COUNT(*) AS client_count,
    SUM(subscribed) AS subscriber_count,
    ROUND(
        100.0 * SUM(subscribed) / COUNT(*),
        2
    ) AS subscription_rate
FROM campaign_segments
GROUP BY campaign_contact_group
ORDER BY
    CASE campaign_contact_group
        WHEN '1 contact' THEN 1
        WHEN '2-3 contacts' THEN 2
        WHEN '4-5 contacts' THEN 3
        WHEN '6-10 contacts' THEN 4
        WHEN '11+ contacts' THEN 5
    END;


-- name: job_performance_ranking
-- Rank job categories and calculate each job's share of subscribers.

WITH job_performance AS (
    SELECT
        job,
        COUNT(*) AS client_count,
        SUM(subscribed) AS subscriber_count,
        ROUND(
            100.0 * SUM(subscribed) / COUNT(*),
            2
        ) AS subscription_rate
    FROM bank_marketing
    GROUP BY job
)
SELECT
    job,
    client_count,
    subscriber_count,
    subscription_rate,
    DENSE_RANK() OVER (
        ORDER BY subscription_rate DESC
    ) AS subscription_rate_rank,
    ROUND(
        100.0 * subscriber_count
        / SUM(subscriber_count) OVER (),
        2
    ) AS share_of_all_subscribers
FROM job_performance
ORDER BY
    subscription_rate_rank,
    subscriber_count DESC;


-- name: scalable_segment_ranking
-- Rank sufficiently large segments across several business dimensions.

WITH segment_performance AS (
    SELECT
        'job' AS feature,
        job AS segment,
        COUNT(*) AS client_count,
        SUM(subscribed) AS subscriber_count,
        ROUND(
            100.0 * SUM(subscribed) / COUNT(*),
            2
        ) AS subscription_rate
    FROM bank_marketing
    GROUP BY job

    UNION ALL

    SELECT
        'contact' AS feature,
        contact AS segment,
        COUNT(*) AS client_count,
        SUM(subscribed) AS subscriber_count,
        ROUND(
            100.0 * SUM(subscribed) / COUNT(*),
            2
        ) AS subscription_rate
    FROM bank_marketing
    GROUP BY contact

    UNION ALL

    SELECT
        'month' AS feature,
        contact_month AS segment,
        COUNT(*) AS client_count,
        SUM(subscribed) AS subscriber_count,
        ROUND(
            100.0 * SUM(subscribed) / COUNT(*),
            2
        ) AS subscription_rate
    FROM bank_marketing
    GROUP BY contact_month

    UNION ALL

    SELECT
        'previous_outcome' AS feature,
        previous_outcome AS segment,
        COUNT(*) AS client_count,
        SUM(subscribed) AS subscriber_count,
        ROUND(
            100.0 * SUM(subscribed) / COUNT(*),
            2
        ) AS subscription_rate
    FROM bank_marketing
    GROUP BY previous_outcome
),
eligible_segments AS (
    SELECT
        feature,
        segment,
        client_count,
        subscriber_count,
        subscription_rate
    FROM segment_performance
    WHERE client_count >= 452
),
ranked_segments AS (
    SELECT
        feature,
        segment,
        client_count,
        subscriber_count,
        subscription_rate,
        ROW_NUMBER() OVER (
            PARTITION BY feature
            ORDER BY
                subscription_rate DESC,
                subscriber_count DESC
        ) AS segment_rank
    FROM eligible_segments
)
SELECT
    feature,
    segment,
    client_count,
    subscriber_count,
    subscription_rate,
    segment_rank
FROM ranked_segments
ORDER BY
    feature,
    segment_rank;
    