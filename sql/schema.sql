-- Өгөгдлийн сангийн бүтэц (SQLite; 2-р сард deploy хийхдээ PostgreSQL руу шилжүүлнэ:
-- TEXT цаг -> TIMESTAMPTZ, INTEGER PRIMARY KEY -> BIGSERIAL).
-- Бүх цаг UTC, ISO-8601 текст ('2024-03-08T13:00:00+00:00'). Лааны цаг = НЭЭЛТИЙН цаг.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS instruments (
    id          INTEGER PRIMARY KEY,
    symbol      TEXT NOT NULL UNIQUE,      -- 'EURUSD'
    pip_size    REAL NOT NULL              -- 0.0001
);

CREATE TABLE IF NOT EXISTS candles (
    instrument_id INTEGER NOT NULL REFERENCES instruments(id),
    timeframe     TEXT    NOT NULL,        -- 'H1'
    ts_utc        TEXT    NOT NULL,
    source        TEXT    NOT NULL,        -- 'dukascopy' | 'histdata' | 'oanda'
    price_side    TEXT    NOT NULL DEFAULT 'BID',
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume        REAL,                    -- tick volume (Dukascopy), HistData-д байхгүй
    is_clean      INTEGER NOT NULL DEFAULT 1,  -- 0 = чанарын шалгалтад унасан
    PRIMARY KEY (instrument_id, timeframe, source, ts_utc)
);

CREATE TABLE IF NOT EXISTS data_quality_issues (
    id         INTEGER PRIMARY KEY,
    source     TEXT NOT NULL,
    ts_utc     TEXT NOT NULL,
    issue      TEXT NOT NULL,              -- missing_hour | spike | flat_bar | ...
    detail     TEXT,
    resolution TEXT                        -- 'kept' | 'dropped' | 'replaced_from_histdata'
);
CREATE INDEX IF NOT EXISTS ix_dq_issue ON data_quality_issues(issue, ts_utc);

CREATE TABLE IF NOT EXISTS econ_events (
    event_id      INTEGER PRIMARY KEY,
    ts_utc        TEXT NOT NULL,
    currency      TEXT NOT NULL,           -- 'USD' | 'EUR'
    impact        TEXT NOT NULL,           -- 'high' | 'medium' | 'low'
    title         TEXT NOT NULL,           -- 'Non-Farm Employment Change'
    actual        REAL, forecast REAL, previous REAL,
    surprise      REAL,                    -- actual - forecast
    surprise_z    REAL,                    -- өмнөх гаргалтуудын std-д хуваасан
    raw_time_text TEXT,                    -- эх файл дахь цаг (шалгахад)
    time_source   TEXT,                    -- original | schedule | learned_year | learned_all
    source        TEXT NOT NULL,           -- 'forexfactory'
    UNIQUE (ts_utc, currency, title)
);
CREATE INDEX IF NOT EXISTS ix_events_ts ON econ_events(ts_utc);

-- Мэдээ ба лааны холбоос (ханш-мэдээг цагаар нэгтгэсэн үр дүн)
CREATE TABLE IF NOT EXISTS candle_events (
    candle_ts_utc TEXT    NOT NULL,
    event_id      INTEGER NOT NULL REFERENCES econ_events(event_id),
    shifted       INTEGER NOT NULL DEFAULT 0,  -- 1 = тухайн цагт лаа байгаагүй, дараагийн лаанд оноосон
    PRIMARY KEY (candle_ts_utc, event_id)
);

-- ===== Дараагийн шатуудад (12-р сар – 2-р сар) ашиглах хүснэгтүүд =====
CREATE TABLE IF NOT EXISTS model_runs (
    run_id      INTEGER PRIMARY KEY,
    created_utc TEXT NOT NULL,
    model       TEXT NOT NULL,             -- 'baseline' | 'lgbm_nonews' | 'lgbm_news' | 'lstm'
    train_from  TEXT, train_to TEXT, test_from TEXT, test_to TEXT,
    params_json TEXT, metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES model_runs(run_id),
    made_at_utc   TEXT NOT NULL,           -- лааны хаалт
    horizon_h     INTEGER NOT NULL,        -- 24
    p_up          REAL NOT NULL,           -- өсөх магадлал
    up_q50 REAL, up_q80 REAL,              -- өсвөл хүрэх түвшин (50%, 80% магадлалтай)
    down_q50 REAL, down_q80 REAL,
    realized_close REAL, realized_high REAL, realized_low REAL  -- хугацаа дуусахад бөглөнө
);
CREATE INDEX IF NOT EXISTS ix_pred_time ON predictions(made_at_utc);
