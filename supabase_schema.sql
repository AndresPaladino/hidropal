-- Esquema HidroPal en Supabase (Postgres).
-- Ejecutar una vez en el SQL Editor de Supabase.

create table if not exists mediciones (
  id          bigint generated always as identity primary key,
  fecha       date        not null,
  nivel       numeric     not null,
  lluvia      numeric     not null default 0,
  extraccion  numeric     not null default 0,
  deleted_at  timestamptz,                   -- papelera si no es null
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- Un unico registro ACTIVO por dia. La papelera (deleted_at no nulo) puede
-- tener fechas repetidas o que coincidan con un activo (datos historicos).
create unique index if not exists mediciones_fecha_activo_uniq
  on mediciones (fecha) where deleted_at is null;

-- La app se conecta con la service_role key (server-side, en st.secrets),
-- que bypassa RLS. Igualmente dejamos RLS habilitado y sin policies para
-- bloquear el acceso desde la anon key publica.
alter table mediciones enable row level security;
