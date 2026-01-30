create table if not exists txn_labels (
  event_id text primary key,
  label int not null,
  label_source text not null,
  labeled_at timestamptz not null default now()
);

create index if not exists txn_labels_labeled_at_idx on txn_labels (labeled_at);
create index if not exists txn_labels_label_idx on txn_labels (label);

create or replace view decision_with_labels as
select
  d.*,
  l.label as true_label,
  l.label_source,
  l.labeled_at
from decision_log d
left join txn_labels l using (event_id);
