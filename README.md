# HidroPal

Dashboard mobile-first para registrar y analizar el pozo de casa: nivel de agua,
lluvia y volumen extraido.

## Arquitectura

- **Streamlit** (mobile-first), modularizado en el paquete `hidropal/`.
- **Supabase (Postgres)** como base de datos (tabla `mediciones`, soft-delete para
  la papelera). Reemplaza el viejo almacenamiento en CSV commiteado a GitHub.
- **Vista publica de solo-lectura**; la edicion se protege con un PIN que se
  recuerda ~1 anio via cookie firmada.
- Respaldo en CSV: boton de descarga + export opcional a una rama de GitHub.

```
app.py                      entrypoint (page config, navegacion, auth gate)
hidropal/
  config.py   constantes y lectura de st.secrets
  db.py       acceso a Supabase (load/upsert/soft_delete/restore/purge/export/backup)
  domain.py   validacion, offset de nivel (-0.17), derivadas, fechas
  auth.py     PIN + cookie de larga duracion
  charts.py   graficas Plotly
  styles.py   CSS mobile-first
  pages_ui/   analisis, cargar, modificar, eliminar, restaurar
supabase_schema.sql          esquema de la tabla
migrate_csv_to_supabase.py   migracion CSV (branch data) -> Supabase
```

## Puesta en marcha (local)

1. `pip install -r requirements.txt`
2. Crear proyecto en [supabase.com](https://supabase.com) y correr `supabase_schema.sql`
   en el SQL Editor.
3. Copiar `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml` y completar
   `url`, `service_key`, `pin_hash` y `cookie_secret`.
4. Migrar los datos historicos: `python migrate_csv_to_supabase.py`
   (lee la branch `data`, fuente de verdad).
5. `streamlit run app.py`

## Deploy

Streamlit Community Cloud. Cargar los mismos secrets en Settings -> Secrets.

## Datos

- Un registro **activo** por dia (indice unico parcial sobre `deleted_at is null`).
- La **papelera** son filas con `deleted_at` no nulo; se restauran o se purgan.
- Al guardar un nivel medido se le resta `0.17 m` (offset de la cinta).
