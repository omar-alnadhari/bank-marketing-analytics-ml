-- Create the SQLite schema for the cleaned Bank Marketing dataset.

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS bank_marketing;

CREATE TABLE bank_marketing (
    campaign_record_id INTEGER PRIMARY KEY AUTOINCREMENT,

    age INTEGER NOT NULL
        CHECK (age > 0),

    job TEXT NOT NULL,
    marital TEXT NOT NULL,
    education TEXT NOT NULL,

    default_status TEXT NOT NULL
        CHECK (default_status IN ('yes', 'no')),

    balance INTEGER NOT NULL,

    housing_loan TEXT NOT NULL
        CHECK (housing_loan IN ('yes', 'no')),

    personal_loan TEXT NOT NULL
        CHECK (personal_loan IN ('yes', 'no')),

    contact TEXT NOT NULL,

    contact_day INTEGER NOT NULL
        CHECK (contact_day BETWEEN 1 AND 31),

    contact_month TEXT NOT NULL,

    call_duration_seconds INTEGER NOT NULL
        CHECK (call_duration_seconds >= 0),

    campaign_contacts INTEGER NOT NULL
        CHECK (campaign_contacts >= 1),

    days_since_previous_contact INTEGER NOT NULL
        CHECK (days_since_previous_contact >= -1),

    previous_contacts INTEGER NOT NULL
        CHECK (previous_contacts >= 0),

    previous_outcome TEXT NOT NULL,

    subscribed INTEGER NOT NULL
        CHECK (subscribed IN (0, 1))
);

CREATE INDEX idx_bank_marketing_subscribed
    ON bank_marketing (subscribed);

CREATE INDEX idx_bank_marketing_job
    ON bank_marketing (job);

CREATE INDEX idx_bank_marketing_month
    ON bank_marketing (contact_month);

CREATE INDEX idx_bank_marketing_previous_outcome
    ON bank_marketing (previous_outcome);