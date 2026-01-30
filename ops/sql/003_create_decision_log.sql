create table if not exists decision_log (
  decision_id bigserial primary key,
  event_id text,
  event_ts timestamptz not null,
  user_id bigint not null,
  merchant_id bigint not null,
  device_id bigint not null,
  ip_id bigint not null,
  amount double precision not null,
  country text not null,
  channel text not null,
  drift_phase int not null,
  final_score double precision not null,
  decision text not null,
  scores jsonb not null,
  fallbacks jsonb not null,
  model_versions jsonb not null,
  features jsonb not null,
  latency_ms jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists decision_log_created_at_idx on decision_log (created_at);
create index if not exists decision_log_event_ts_idx on decision_log (event_ts);
create index if not exists decision_log_country_idx on decision_log (country);
create index if not exists decision_log_decision_idx on decision_log (decision);
create index if not exists decision_log_user_created_idx on decision_log (user_id, created_at);
