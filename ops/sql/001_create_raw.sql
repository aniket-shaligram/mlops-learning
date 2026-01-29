create table if not exists txn_raw (
  event_id text primary key,
  event_type text not null,
  event_ts timestamptz not null,
  user_id bigint not null,
  merchant_id bigint not null,
  device_id bigint not null,
  ip_id bigint not null,
  amount double precision not null,
  currency text not null,
  country text not null,
  channel text not null,
  drift_phase int not null default 0,
  payload jsonb not null,
  ingested_at timestamptz not null default now()
);

create index if not exists txn_raw_event_ts_idx on txn_raw (event_ts);
create index if not exists txn_raw_user_id_idx on txn_raw (user_id);
