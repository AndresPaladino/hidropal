import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import date
import os
import locale
import hashlib
import base64
import io
import requests

# -------------------------
# Locale (opcional)
# -------------------------
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')      # Linux/Mac
except:
    try:
        locale.setlocale(locale.LC_TIME, 'Spanish_Spain.1252')  # Windows
    except:
        pass  # Usamos %d/%m/%Y numérico igualmente

CSV_FILE = "datos_pozo.csv"
TRASH_FILE = "datos_pozo_borrados.csv"
st.set_page_config(page_title="Hidropal", page_icon="logo_pozo.svg", initial_sidebar_state="collapsed", layout="wide")

# -------------------------
# Almacenamiento remoto (GitHub) opcional
# -------------------------
def _gh_cfg():
    """Lee configuración de GitHub desde st.secrets. Retorna dict o None si no está configurado."""
    try:
        # Permitir dos formatos: secretos planos o anidados en 'github'
        sec = st.secrets.get("github", st.secrets)
        token = sec.get("GITHUB_TOKEN")
        repo = sec.get("GITHUB_REPO")  # formato: 'owner/repo'
        branch = sec.get("GITHUB_BRANCH", "master")
        data_path = sec.get("GITHUB_DATA_PATH", CSV_FILE)
        trash_path = sec.get("GITHUB_TRASH_PATH", TRASH_FILE)
        if token and repo:
            return {
                "token": token,
                "repo": repo,
                "branch": branch,
                "data_path": data_path,
                "trash_path": trash_path,
            }
    except Exception:
        pass
    return None

def gh_enabled() -> bool:
    return _gh_cfg() is not None

def _gh_headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "hidropal-app"
    }

def _gh_contents_url(repo: str, path: str) -> str:
    owner, name = repo.split("/")
    return f"https://api.github.com/repos/{owner}/{name}/contents/{path}"

def gh_get_file(path: str):
    """Obtiene contenido (str) y sha del archivo en GitHub. Retorna (None, None) si no existe."""
    cfg = _gh_cfg()
    if not cfg:
        return None, None
    url = _gh_contents_url(cfg["repo"], path)
    params = {"ref": cfg["branch"]}
    r = requests.get(url, headers=_gh_headers(cfg["token"]), params=params, timeout=20)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    data = r.json()
    content_b64 = data.get("content", "")
    if data.get("encoding") == "base64" and content_b64:
        content_bytes = base64.b64decode(content_b64)
        return content_bytes.decode("utf-8"), data.get("sha")
    return None, data.get("sha")

def gh_put_file(path: str, content_str: str, message: str, sha: str | None = None):
    """Crea/actualiza archivo en GitHub (Contents API)."""
    cfg = _gh_cfg()
    if not cfg:
        return False
    url = _gh_contents_url(cfg["repo"], path)
    payload = {
        "message": message,
        "content": base64.b64encode(content_str.encode("utf-8")).decode("ascii"),
        "branch": cfg["branch"],
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_gh_headers(cfg["token"]), json=payload, timeout=20)
    if r.status_code in (200, 201):
        return True
    # En caso de conflicto por SHA, devolver False para que el caller reintente
    return False

def destination_exists(path: str) -> bool:
    if gh_enabled():
        content, _ = gh_get_file(path)
        return content is not None
    return os.path.exists(path)

def read_csv_from_destination(path: str) -> pd.DataFrame:
    if gh_enabled():
        content, _ = gh_get_file(path)
        if content is None:
            raise FileNotFoundError(path)
        return pd.read_csv(io.StringIO(content))
    return pd.read_csv(path)

def write_csv_to_destination(df: pd.DataFrame, path: str, commit_message: str) -> bool:
    if gh_enabled():
        # Leer SHA actual (si existe) para update seguro
        _, sha = gh_get_file(path)
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        ok = gh_put_file(path, csv_buf.getvalue(), commit_message, sha)
        return ok
    else:
        df.to_csv(path, index=False)
        return True

# -------------------------
# Paleta de colores consistente
# -------------------------
COLOR_PALETTE = {
    "NIVEL": "#d62728",          # Rojo
    "LLUVIA": "#000fff",         # Azul
    "EXTRACCION": "#2ca02c",     # Verde
    "VARIACION_NIVEL": "#ff7f0e", # Naranja
    "LLUVIA_ACUM_7D": "#9467bd"  # Púrpura
}

# -------------------------
# Normalización de columnas
# -------------------------
RENAME_MAP = {
    "FECHA": "FECHA",
    "NIVEL DE AGUA (MTS.)": "NIVEL",
    "LLUVIA CAIDA (MM)": "LLUVIA",
    "VOLUMEN EXTRAIDO (LTS.)": "EXTRACCION",
    # Si ya vienen cortas:
    "NIVEL": "NIVEL",
    "LLUVIA": "LLUVIA",
    "EXTRACCION": "EXTRACCION",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_upper = [c.strip().upper() for c in df.columns]
    df.columns = cols_upper
    df = df.rename(columns={c: RENAME_MAP.get(c, c) for c in df.columns})
    # Asegurar columnas mínimas
    for c in ["FECHA", "NIVEL", "LLUVIA", "EXTRACCION"]:
        if c not in df.columns:
            df[c] = np.nan
    return df[["FECHA", "NIVEL", "LLUVIA", "EXTRACCION"]]

# -------------------------
# Utilidades de fechas
# -------------------------
DATE_OUT_FMT = "%d/%m/%Y"
DATE_TICK_FMTTER = mdates.DateFormatter('%d/%m/%Y')

def ensure_datetime_es(s: pd.Series) -> pd.Series:
    """Convierte a datetime interpretando día/mes/año."""
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def to_es_date_str(s: pd.Series) -> pd.Series:
    """Convierte datetime a string dd/mm/aaaa (sin tocar strings ya formateados)."""
    if np.issubdtype(s.dtype, np.datetime64):
        return s.dt.strftime(DATE_OUT_FMT)
    parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")
    out = s.astype("object").copy()
    mask = parsed.notna()
    out[mask] = parsed[mask].dt.strftime(DATE_OUT_FMT)
    return out

def apply_date_formatter(ax):
    ax.xaxis.set_major_formatter(DATE_TICK_FMTTER)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

# -------------------------
# IDs estables por fila
# -------------------------
def row_fingerprint(fecha_str: str, nivel, lluvia, extraccion) -> str:
    """
    Genera un ID estable SHA1 a partir de los cuatro campos normalizados.
    fecha_str debe venir en dd/mm/yyyy.
    """
    def norm_num(x):
        # Representación consistente para flotantes
        try:
            return f"{float(x):.6f}"
        except:
            return "nan"
    base = f"{fecha_str}|{norm_num(nivel)}|{norm_num(lluvia)}|{norm_num(extraccion)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

def ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura columna ID calculada en base a FECHA(dd/mm/yyyy),NIVEL,LLUVIA,EXTRACCION."""
    if df.empty:
        return df.assign(ID=pd.Series(dtype="object"))
    tmp = df.copy()
    # FECHA a string dd/mm/aaaa
    fecha_str = to_es_date_str(ensure_datetime_es(tmp["FECHA"]))
    ids = []
    for f, n, l, e in zip(fecha_str, tmp["NIVEL"], tmp["LLUVIA"], tmp["EXTRACCION"]):
        ids.append(row_fingerprint(str(f), n, l, e))
    tmp["ID"] = ids
    # Evitar colisiones improbables duplicando ID (mantener primero)
    tmp = tmp.drop_duplicates(subset=["ID"], keep="first")
    return tmp

# -------------------------
# Carga / guardado principal
# -------------------------
def load_data() -> pd.DataFrame:
    try:
        df = read_csv_from_destination(CSV_FILE)  # CSV con comas
        df = normalize_columns(df)
        if "FECHA" in df.columns:
            df["FECHA"] = ensure_datetime_es(df["FECHA"])
        df = ensure_ids(df)
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["FECHA", "NIVEL", "LLUVIA", "EXTRACCION", "ID"])

def save_data(new_row: dict):
    df = load_data()
    # Normalizar fecha a dd/mm/aaaa
    fecha_fmt = ensure_datetime_es(pd.Series([new_row.get("FECHA")])).dt.strftime(DATE_OUT_FMT).iloc[0]
    row = {
        "FECHA": fecha_fmt,
        "NIVEL": new_row.get("NIVEL"),
        "LLUVIA": new_row.get("LLUVIA"),
        "EXTRACCION": new_row.get("EXTRACCION"),
    }
    row["ID"] = row_fingerprint(row["FECHA"], row["NIVEL"], row["LLUVIA"], row["EXTRACCION"])
    # FECHA como string en el CSV
    if "FECHA" in df.columns:
        df["FECHA"] = ensure_datetime_es(df["FECHA"]).dt.strftime(DATE_OUT_FMT)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    write_csv_to_destination(df, CSV_FILE, commit_message=f"chore(data): add row {row['FECHA']}")

def overwrite_data(df: pd.DataFrame):
    """Guarda un DataFrame completo aplicando formato dd/mm/aaaa a FECHA."""
    if "FECHA" in df.columns:
        df = df.copy()
        df["FECHA"] = ensure_datetime_es(df["FECHA"]).dt.strftime(DATE_OUT_FMT)
    df = ensure_ids(df)
    write_csv_to_destination(df, CSV_FILE, commit_message="chore(data): overwrite data")

# -------------------------
# Papelera (borrados)
# -------------------------
def load_trash() -> pd.DataFrame:
    try:
        df = read_csv_from_destination(TRASH_FILE)
        df = normalize_columns(df)
        # ID puede venir o no; si no, lo recalculamos
        if "FECHA" in df.columns:
            # mantener FECHA como string en papelera
            pass
        if "ID" not in df.columns:
            df = ensure_ids(df)
        return df[["FECHA", "NIVEL", "LLUVIA", "EXTRACCION", "ID"]]
    except FileNotFoundError:
        return pd.DataFrame(columns=["FECHA", "NIVEL", "LLUVIA", "EXTRACCION", "ID"])

def append_to_trash(rows: pd.DataFrame):
    trash = load_trash()
    # Asegurar FECHA string
    rows = rows.copy()
    rows["FECHA"] = to_es_date_str(ensure_datetime_es(rows["FECHA"]))
    rows = ensure_ids(rows)
    trash = pd.concat([trash, rows], ignore_index=True)
    # Evitar duplicados por ID en la papelera
    trash = trash.drop_duplicates(subset=["ID"], keep="first")
    write_csv_to_destination(trash, TRASH_FILE, commit_message="chore(trash): append rows")

def remove_from_trash_by_id(row_id: str):
    trash = load_trash()
    trash = trash[trash["ID"] != row_id].copy()
    write_csv_to_destination(trash, TRASH_FILE, commit_message="chore(trash): remove row")

# -------------------------
# Validación de datos de entrada
# -------------------------
def validate_input_data(fecha, nivel, lluvia, extraccion):
    """
    Valida y limpia datos de entrada.
    Retorna: (datos_limpios_dict, lista_errores)
    """
    errors = []
    
    # Limpiar y validar FECHA
    if fecha is None:
        fecha = date.today()
    if fecha > date.today():
        errors.append("📅 La fecha no puede ser futura")
    
    # Validar NIVEL (obligatorio, > 0)
    if nivel is None:
        errors.append("💧 El nivel no puede ser vacío")
    elif nivel <= 0:
        errors.append("💧 El nivel debe ser mayor a 0")
    
    # Limpiar y validar LLUVIA
    if lluvia is None:
        lluvia = 0.0
    elif lluvia < 0:
        errors.append("🌧️ Lluvia no puede ser negativa")
    
    # Limpiar y validar EXTRACCIÓN
    if extraccion is None:
        extraccion = 0.0
    elif extraccion < 0:
        errors.append("🚰 Extracción no puede ser negativa")
    
    cleaned_data = {
        "FECHA": fecha,
        "NIVEL": nivel if nivel is not None and nivel > 0 else None,
        "LLUVIA": lluvia,
        "EXTRACCION": extraccion
    }
    
    return cleaned_data, errors

def restore_row_by_id(row_id: str):
    trash = load_trash()
    if trash.empty:
        return False, "Papelera vacía."
    sel = trash[trash["ID"] == row_id]
    if sel.empty:
        return False, "No se encontró el registro en papelera."
    row = sel.iloc[0]
    # Cargar datos actuales
    df = load_data()
    # FECHA del CSV debe ser string
    new_row = {
        "FECHA": str(row["FECHA"]),
        "NIVEL": row["NIVEL"],
        "LLUVIA": row["LLUVIA"],
        "EXTRACCION": row["EXTRACCION"],
        "ID": row["ID"],
    }
    # Evitar duplicados por ID
    if not df[df["ID"] == new_row["ID"]].empty:
        # Ya existe en principal; solo quitamos de papelera
        remove_from_trash_by_id(new_row["ID"])
        return True, "El registro ya estaba en los datos; se quitó de la papelera."
    # Agregar y guardar
    # Asegurar FECHA string en CSV
    if "FECHA" in df.columns:
        df["FECHA"] = ensure_datetime_es(df["FECHA"]).dt.strftime(DATE_OUT_FMT)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    write_csv_to_destination(df, CSV_FILE, commit_message="chore(data): restore row from trash")
    # Quitar de papelera
    remove_from_trash_by_id(new_row["ID"])
    return True, "Registro restaurado."

# -------------------------
# UI
# -------------------------
# Leer el SVG desde archivo
with open("logo_grande.svg", "r") as f:
    svg_icon = f.read()

# Mostrar SVG + texto al lado
st.markdown(
    f"""
    <style>
      .hidropal-hero {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: clamp(8px, 2vw, 20px);
        margin: clamp(8px, 3vw, 24px) auto;
        padding: 0 clamp(8px, 2vw, 24px);
        width: fit-content;                 /* evita que el contenedor “tire” a un lado */
      }}

      .hidropal-logo {{
        width: clamp(40px, 12vw, 96px);
        height: clamp(40px, 12vw, 96px);
        display: block;                     /* que el svg no sea inline */
        line-height: 0;
        margin: 0 auto;                     /* centrado perfecto si cambia a grid */
      }}
      .hidropal-logo svg {{
        width: 100%;
        height: 100%;
        display: block;
      }}

      .hidropal-title {{
        font-size: clamp(1.4rem, 4vw, 2.4rem);
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
        text-align: center;                 /* por si hay varias líneas */
      }}

      /* En pantallas chicas usamos GRID para centrar ambos por geometría */
      @media (max-width: 640px) {{
        .hidropal-logo {{
            transform: translateX(-10px);
        }}
        .hidropal-hero {{
          display: grid;
          grid-auto-flow: row;
          justify-content: center;
          justify-items: center;            /* centra cada ítem (logo y título) */
          align-items: center;
          row-gap: clamp(6px, 2.5vw, 12px);
          width: 100%;                      /* ocupa ancho y centra internamente */
        }}
      }}
      
    </style>

    <div class="hidropal-hero">
      <div class="hidropal-logo">{svg_icon}</div>
      <h1 class="hidropal-title">HidroPal</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Estado para “deshacer último borrado”
if "last_deleted_rows" not in st.session_state:
    st.session_state["last_deleted_rows"] = None  # DataFrame con las filas borradas

# --- Config inicial: subir CSV ---
st.sidebar.image("logo_grande.svg", width=120,)
if not destination_exists(CSV_FILE):
    st.sidebar.info("No existe un archivo de datos. Subí un CSV inicial (con comas).")
    uploaded_file = st.sidebar.file_uploader("Subir CSV inicial", type=["csv"])
    if uploaded_file:
        df_init = pd.read_csv(uploaded_file)  # comas
        df_init = normalize_columns(df_init)
        if "FECHA" in df_init.columns:
            df_init["FECHA"] = ensure_datetime_es(df_init["FECHA"]).dt.strftime(DATE_OUT_FMT)
        df_init = ensure_ids(df_init)
        write_csv_to_destination(df_init, CSV_FILE, commit_message="chore(data): initial upload")
        st.sidebar.success("✅ Archivo inicial cargado correctamente.")
else:
    st.sidebar.success("Usando archivo existente")

# -------------------------
# NAVBAR con tabs
# -------------------------
tab_datos, tab_analisis = st.tabs(["🧾 Datos", "📈 Análisis"])

# ====== TAB DATOS ======
with tab_datos:
    subtab_cargar, subtab_modificar, subtab_eliminar, subtab_restaurar = st.tabs(["➕ Cargar", "✏️ Modificar", "🗑️ Eliminar", "♻️ Restaurar"])

    # ---- Cargar ----
    with subtab_cargar:
        st.subheader("Cargar nueva medición")
        
        # Inicializar flag para resetear después de guardar
        if "reset_after_save" not in st.session_state:
            st.session_state.reset_after_save = False
        
        # Si acabamos de guardar, resetear todo
        if st.session_state.reset_after_save:
            st.session_state.fecha_cargar = None
            st.session_state.nivel_input = None
            st.session_state.lluvia_input = None
            st.session_state.extraccion_input = None
            st.session_state.current_fecha_key = None
            st.session_state.reset_after_save = False
            st.success("✨ Listo para cargar nueva medición")
        
        # Pre-cargar datos existentes si hay registro para la fecha seleccionada
        df_all = load_data()
        df_all["LLUVIA"] = df_all["LLUVIA"].fillna(0)
        
        # Selector de fecha - usar session_state para valor por defecto
        default_fecha = st.session_state.get("fecha_cargar", date.today())
        fecha = st.date_input(
            "Fecha", 
            value=default_fecha, 
            max_value=date.today(), 
            format="DD/MM/YYYY",
            key="fecha_cargar"
        )
        
        # Buscar si existe un registro para esta fecha
        if fecha is not None:
            fecha_dt = pd.to_datetime(fecha)
            existing_row = df_all[df_all["FECHA"].dt.date == fecha_dt.date()]
            is_editing = not existing_row.empty
            
            # Inicializar o actualizar valores cuando cambia la fecha
            fecha_key = fecha.strftime("%Y%m%d")
            if "current_fecha_key" not in st.session_state or st.session_state.current_fecha_key != fecha_key:
                st.session_state.current_fecha_key = fecha_key
                
                if not existing_row.empty:
                    st.session_state.nivel_input = float(existing_row.iloc[0]["NIVEL"])
                    st.session_state.lluvia_input = float(existing_row.iloc[0]["LLUVIA"])
                    st.session_state.extraccion_input = float(existing_row.iloc[0]["EXTRACCION"])
                else:
                    # Usar None para campos vacíos
                    st.session_state.nivel_input = None
                    st.session_state.lluvia_input = None
                    st.session_state.extraccion_input = None
            
            # Mensaje diferenciado según si ya existen datos
            if is_editing:
                # Mostrar card con datos actuales
                st.success(f"✅ Datos del {fecha.strftime('%d/%m/%Y')} ya están guardados")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💧 Nivel", f"{float(existing_row.iloc[0]['NIVEL']):.2f} m")
                with col2:
                    st.metric("🌧️ Lluvia", f"{float(existing_row.iloc[0]['LLUVIA']):.0f} mm")
                with col3:
                    st.metric("🚰 Extracción", f"{float(existing_row.iloc[0]['EXTRACCION']):.0f} lts")
                
                st.info("💡 Para cargar datos de otro día, selecciona una fecha diferente arriba")
                
                # Botón para ir a modificar (placeholder por ahora)
                if st.button("✏️ Modificar estos datos", type="secondary"):
                    # Preseleccionar este registro en la pestaña "Modificar"
                    try:
                        st.session_state.registro_a_modificar = str(existing_row.iloc[0]["ID"]) if "ID" in existing_row.columns else None
                    except Exception:
                        st.session_state.registro_a_modificar = None
                    st.toast("✏️ Listo: cambia a la pestaña 'Modificar' para editar este registro.")
                
            else:
                # No hay datos, permitir carga
                button_text = "💾 Guardar"
                # Inicializar los valores en session_state si no existen
                if "nivel_input" not in st.session_state:
                    st.session_state.nivel_input = 0.0
                if "lluvia_input" not in st.session_state:
                    st.session_state.lluvia_input = 0.0
                if "extraccion_input" not in st.session_state:
                    st.session_state.extraccion_input = 0.0
                
                # Inputs de datos con keys vinculadas a session_state
                nivel = st.number_input("Nivel del agua (m)", value=st.session_state.nivel_input, format="%.2f", key="nivel_input")
                lluvia = st.number_input("Lluvia caída (mm)", value=st.session_state.lluvia_input, format="%.2f", key="lluvia_input")
                extraccion = st.number_input("Volumen extraído (lts)", value=st.session_state.extraccion_input, format="%.2f", key="extraccion_input")
                
                # Botón de guardar
                submitted = st.button(button_text, type="primary")

                if submitted:
                    # Validar datos antes de guardar
                    cleaned_data, errors = validate_input_data(fecha, nivel, lluvia, extraccion)
                    
                    if errors:
                        # Mostrar errores de forma compacta
                        error_msg = " • ".join(errors)
                        st.error(f"⚠️ {error_msg}")
                    else:
                        # Datos válidos, guardar
                        save_data({
                            "FECHA": cleaned_data["FECHA"], 
                            "NIVEL": cleaned_data["NIVEL"] - 0.17, 
                            "LLUVIA": cleaned_data["LLUVIA"], 
                            "EXTRACCION": cleaned_data["EXTRACCION"]
                        })
                        st.toast("✅ Datos guardados correctamente", icon="✅")
                        st.balloons()
                        # Marcar para resetear en próximo rerun
                        st.session_state.reset_after_save = True
                        st.rerun()
        else:
            st.info("📅 Selecciona una fecha para comenzar")


        # Vista rápida de los últimos registros
        df_head = load_data().sort_values("FECHA", ascending=False).head(10)
        if not df_head.empty:
            df_head_disp = df_head.drop(columns=["ID"]).copy()
            df_head_disp["FECHA"] = to_es_date_str(df_head_disp["FECHA"])
            st.caption("Últimos 10 registros:")
            st.dataframe(df_head_disp, hide_index=True, use_container_width=True)

    # ---- Modificar ----
    with subtab_modificar:
        st.subheader("Modificar una medición existente")
        
        df_all = load_data().sort_values("FECHA", ascending=False)
        
        if df_all.empty:
            st.info("⚠️ No hay datos para modificar. Carga datos primero en la pestaña 'Cargar'.")
        else:
            # Construir selector con formato legible
            df_disp = df_all.copy()
            df_disp["FECHA_STR"] = to_es_date_str(df_disp["FECHA"])
            opciones = [
                f"{row['FECHA_STR']} | Nivel={row['NIVEL']:.2f}m | Lluvia={row['LLUVIA']:.0f}mm | Extracción={row['EXTRACCION']:.0f}lts"
                for _, row in df_disp.iterrows()
            ]
            
            # Mapeo de opción a ID y viceversa
            opciones_to_id = {
                f"{row['FECHA_STR']} | Nivel={row['NIVEL']:.2f}m | Lluvia={row['LLUVIA']:.0f}mm | Extracción={row['EXTRACCION']:.0f}lts": row['ID']
                for _, row in df_disp.iterrows()
            }
            id_to_opciones = {v: k for k, v in opciones_to_id.items()}
            
            # Detectar si venimos de "Cargar" con un registro pre-seleccionado
            default_index = None
            if "registro_a_modificar" in st.session_state and st.session_state.registro_a_modificar:
                id_preseleccionado = st.session_state.registro_a_modificar
                if id_preseleccionado in id_to_opciones:
                    opcion_preseleccionada = id_to_opciones[id_preseleccionado]
                    if opcion_preseleccionada in opciones:
                        # Fijar el valor del selectbox directamente para respetar la selección
                        st.session_state["modificar_selectbox"] = opcion_preseleccionada
                        default_index = opciones.index(opcion_preseleccionada)
                # Limpiar el flag después de usarlo
                st.session_state.registro_a_modificar = None
            
            seleccion = st.selectbox(
                "Selecciona el registro a modificar:", 
                opciones, 
                index=default_index,
                placeholder="Selecciona un registro",
                key="modificar_selectbox"
            )
            
            if seleccion:
                # Obtener el registro seleccionado
                sel_id = opciones_to_id[seleccion]
                registro = df_all[df_all["ID"] == sel_id].iloc[0]
                
                # Mostrar card con datos actuales
                st.info(f"📝 Modificando registro del {to_es_date_str(pd.Series([registro['FECHA']])).iloc[0]}")
                
                # Inputs con valores pre-cargados
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    nuevo_nivel = st.number_input(
                        "Nivel del agua (m)", 
                        value=float(registro["NIVEL"]),
                        format="%.2f",
                        key="modificar_nivel"
                    )
                    st.caption("Si no cambiás el nivel, se conserva igual. Si ingresás un valor medido nuevo, se aplicará ajuste +0.17 al guardar.")
                
                with col2:
                    nueva_lluvia = st.number_input(
                        "Lluvia caída (mm)", 
                        value=None if registro["LLUVIA"] == 0 else float(registro["LLUVIA"])    ,
                        format="%.2f",
                        key="modificar_lluvia"
                    )
                
                with col3:
                    nueva_extraccion = st.number_input(
                        "Volumen extraído (lts)", 
                        value=None if registro["EXTRACCION"] == 0 else float(registro["EXTRACCION"]),
                        format="%.2f",
                        key="modificar_extraccion"
                    )
                
                # Mantener fecha original (no se permite modificarla)
                fecha_actual = pd.to_datetime(registro["FECHA"]).date()
                
                # Botón de modificar
                if st.button("💾 Guardar cambios", type="primary", use_container_width=True):
                    # Validar datos (fecha fija: no se permite modificarla)
                    cleaned_data, errors = validate_input_data(
                        fecha_actual,
                        nuevo_nivel,
                        nueva_lluvia,
                        nueva_extraccion
                    )
                    if errors:
                        error_msg = " • ".join(errors)
                        st.error(f"⚠️ {error_msg}")
                    else:
                        # Mantener misma fecha, solo actualizar valores
                        # Ajuste de nivel solo si el usuario lo cambió
                        nivel_original = float(registro["NIVEL"])
                        nivel_input = float(cleaned_data["NIVEL"]) if cleaned_data["NIVEL"] is not None else nivel_original
                        cambio_nivel = not np.isclose(nivel_input, nivel_original, atol=1e-6)
                        nivel_final = (nivel_input - 0.17) if cambio_nivel else nivel_original
                        df_all.loc[df_all["ID"] == sel_id, "NIVEL"] = nivel_final
                        df_all.loc[df_all["ID"] == sel_id, "LLUVIA"] = cleaned_data["LLUVIA"]
                        df_all.loc[df_all["ID"] == sel_id, "EXTRACCION"] = cleaned_data["EXTRACCION"]
                        overwrite_data(df_all)
                        st.toast("✅ Registro modificado correctamente", icon="✅")
                        st.rerun()

    # ---- Eliminar ----
    with subtab_eliminar:
        st.subheader("Eliminar una medición (se mueve a Papelera)")
        df_current = load_data().sort_values("FECHA")
        if not df_current.empty:
            # Construir etiquetas legibles
            df_disp = df_current.copy()
            df_disp["FECHA_STR"] = to_es_date_str(df_disp["FECHA"])
            opciones = [
                f"{row['FECHA_STR']} | Nivel={row['NIVEL']} | Lluvia={row['LLUVIA']} | Extracción={row['EXTRACCION']} - ID={row['ID']}"
                for _, row in df_disp.iterrows()
            ]

            seleccion = st.selectbox("Selecciona el registro a eliminar:", opciones, index=None, label_visibility="collapsed",placeholder="Selecciona un registro") if opciones else None

            col1, col2 = st.columns([1,1])
            with col1:
                eliminar = st.button("Eliminar registro", type="primary", disabled=(seleccion is None))
            with col2:
                deshacer = st.button("Deshacer último borrado", type="secondary")

            if eliminar:
                if seleccion is None:
                    st.error("⚠️ Debes seleccionar un registro primero.")
                else:
                    # Extraer el ID de la selección (formato: "... - ID=ABC123")
                    sel_id = seleccion.split(" - ID=")[1].strip()
                    # Filas a borrar
                    rows_to_delete = df_current[df_current["ID"] == sel_id]
                    if rows_to_delete.empty:
                        st.error("No se encontró el registro seleccionado.")
                    else:
                        # Guardar para deshacer
                        st.session_state["last_deleted_rows"] = rows_to_delete.copy()
                        # Mover a papelera
                        append_to_trash(rows_to_delete)
                        # Eliminar de principal
                        df_new = df_current[df_current["ID"] != sel_id].reset_index(drop=True)
                        overwrite_data(df_new)
                        st.toast("Registro movido a Papelera", icon="🗑️")
                        st.rerun()

            if deshacer:
                last = st.session_state.get("last_deleted_rows")
                if last is None or last.empty:
                    st.info("No hay un borrado reciente para deshacer.")
                else:
                    # Restaurar desde la copia de sesión (no depende de archivo papelera)
                    df_now = load_data()
                    # Asegurar FECHA string antes de guardar
                    if "FECHA" in df_now.columns:
                        df_now["FECHA"] = ensure_datetime_es(df_now["FECHA"]).dt.strftime(DATE_OUT_FMT)
                    # Evitar duplicados por ID
                    ids_now = set(df_now["ID"]) if "ID" in df_now.columns else set()
                    rows_add = last[~last["ID"].isin(ids_now)].copy()
                    if not rows_add.empty:
                        # FECHA string
                        rows_add["FECHA"] = to_es_date_str(ensure_datetime_es(rows_add["FECHA"]))
                        rows_add = ensure_ids(rows_add)
                        df_now = pd.concat([df_now, rows_add], ignore_index=True)
                        df_now.to_csv(CSV_FILE, index=False)
                        # Quitar esos IDs de la papelera si existen
                        for rid in rows_add["ID"].tolist():
                            remove_from_trash_by_id(rid)
                        st.toast("Se deshizo el último borrado", icon="↩️")
                    else:
                        st.info("Ese registro ya está presente; no se realizó ninguna acción.")
                    # Limpiar buffer
                    st.session_state["last_deleted_rows"] = None
                    st.rerun()
        else:
            st.info("No hay datos para eliminar.")

    # ---- Restaurar (desde papelera persistente) ----
    with subtab_restaurar:
        st.subheader("Restaurar desde Papelera")
        trash = load_trash()
        if trash.empty:
            st.info("La Papelera está vacía.")
        else:
            # Mostrar selector
            def label_trash_row(r):
                return f"{r['ID']} - {r['FECHA']} | Nivel={r['NIVEL']} | Lluvia={r['LLUVIA']} | Extracción={r['EXTRACCION']}"
            opciones_trash = [label_trash_row(row) for _, row in trash.iterrows()]
            seleccion_trash = st.selectbox("Seleccioná un registro de la Papelera:", opciones_trash)
            colr1, colr2 = st.columns([1,3])
            with colr1:
                restore_btn = st.button("Restaurar", type="primary")
            with colr2:
                purge_btn = st.button("Vaciar Papelera (borra definitivamente)")

            if restore_btn and seleccion_trash:
                sel_id = seleccion_trash.split(" - ")[0].strip()
                ok, msg = restore_row_by_id(sel_id)
                if ok:
                    st.toast(f"{msg}", icon="✅")
                    st.rerun()
                else:
                    st.error(msg)

            if purge_btn:
                # Vaciar papelera completa
                pd.DataFrame(columns=["FECHA","NIVEL","LLUVIA","EXTRACCION","ID"]).to_csv(TRASH_FILE, index=False)
                st.toast("Papelera vaciada", icon="🧹")
                st.rerun()

# ====== TAB ANÁLISIS ======
with tab_analisis:
    df = load_data()

    if not df.empty:
        # Derivadas
        df = df.sort_values("FECHA")
        df["VARIACION_NIVEL"] = (-df["NIVEL"]).diff()
        df["LLUVIA_ACUM_7D"] = df["LLUVIA"].rolling(window=7).sum()

        # Vista de datos (FECHA en dd/mm/aaaa)
        st.subheader("Vista de datos")
        df_display = df.drop(columns=["ID"]).copy()
        df_display["FECHA"] = to_es_date_str(df_display["FECHA"])
        st.dataframe(df_display, hide_index=True, use_container_width=True)

       
        # --- Serie temporal
        st.subheader("Serie temporal")
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.patch.set_alpha(0.0)
        axes[0].plot(df['FECHA'], 6.5 - df['NIVEL'], marker='o', color=COLOR_PALETTE["NIVEL"])
        axes[0].set_title('Nivel de Agua')
        axes[0].set_ylabel('Nivel (m)')
        axes[0].grid(True)

        axes[1].plot(df['FECHA'], df['LLUVIA'], marker='o', color=COLOR_PALETTE["LLUVIA"])
        axes[1].set_title('Lluvia Caída')
        axes[1].set_ylabel('mm')
        axes[1].grid(True)

        axes[2].bar(df['FECHA'], df['EXTRACCION'],  color=COLOR_PALETTE["EXTRACCION"])
        axes[2].set_title('Volumen Extraído')
        axes[2].set_ylabel('Litros')
        axes[2].set_xlabel('Fecha')
        
        axes[2].grid(True)

        for ax in axes:
            apply_date_formatter(ax)
            ax.patch.set_alpha(0.0)
        #fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Dashboard")
        fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(df['FECHA'], df['NIVEL'], marker='o', color=COLOR_PALETTE["NIVEL"])
        axes[0].invert_yaxis()
        axes[0].set_title('Nivel de Agua')
        axes[0].grid(True)

        axes[1].plot(df['FECHA'], df['LLUVIA'], marker='o', color=COLOR_PALETTE["LLUVIA"])
        axes[1].set_title('Lluvia Caída')
        axes[1].set_ylabel('mm')
        axes[1].grid(True)
        
        axes[2].bar(df['FECHA'], df['VARIACION_NIVEL'], color=COLOR_PALETTE["VARIACION_NIVEL"])
        axes[2].axhline(0, color='k', linestyle='--')
        axes[2].set_title('Variación del Nivel')
        axes[2].grid(True)

        axes[3].plot(df['FECHA'], df['LLUVIA_ACUM_7D'], marker='o', color=COLOR_PALETTE["LLUVIA_ACUM_7D"])
        axes[3].set_title('Lluvia acumulada 7 días')
        axes[3].grid(True)

        axes[4].bar(df['FECHA'], df['EXTRACCION'], color=COLOR_PALETTE["EXTRACCION"])
        axes[4].set_title('Volumen extraído')
        axes[4].set_xlabel('Fecha')
        axes[4].grid(True)

        for ax in axes:
            apply_date_formatter(ax)

        #fig.tight_layout()
        st.pyplot(fig)

        # --- multi select superpuestas
        st.subheader("Comparación de tendencias")
        variables = {
            "Nivel": 6.5 - df["NIVEL"],
            "Lluvia": df["LLUVIA"],
            "Extracción": df["EXTRACCION"],
            "Variación de nivel": df["VARIACION_NIVEL"],
            "Lluvia Acumulada (7 dias)": df["LLUVIA_ACUM_7D"]
        }
        seleccion = st.multiselect("Variables a comparar", list(variables.keys()),placeholder="Selecciona una o más variables", default=["Nivel","Lluvia","Extracción"])

        fig, ax = plt.subplots(figsize=(10,6))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        # Mapeo de variables a colores
        var_colors = {
            "Nivel": COLOR_PALETTE["NIVEL"],
            "Lluvia": COLOR_PALETTE["LLUVIA"],
            "Extracción": COLOR_PALETTE["EXTRACCION"],
            "Variación de nivel": COLOR_PALETTE["VARIACION_NIVEL"],
            "Lluvia Acumulada (7 dias)": COLOR_PALETTE["LLUVIA_ACUM_7D"]
        }
        for var in seleccion:
            serie = variables[var]
            serie_norm = (serie - serie.min()) / (serie.max() - serie.min())
            ax.plot(df["FECHA"], serie_norm, label=var, marker="o", color=var_colors[var])

        apply_date_formatter(ax)
        ax.legend()
        st.pyplot(fig)

        # --- Variación vs lluvia acumulada
        st.subheader("Variación del nivel vs Lluvia acumulada (7d)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['LLUVIA_ACUM_7D'], df['VARIACION_NIVEL'], color=COLOR_PALETTE["VARIACION_NIVEL"], alpha=0.6)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel("Lluvia acumulada (mm)")
        ax.set_ylabel("ΔNivel (m)")
        ax.grid(True)
        st.pyplot(fig)

        # --- Variación vs extracción
        st.subheader("Variación del nivel vs Volumen extraído")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['EXTRACCION'], df['VARIACION_NIVEL'], color=COLOR_PALETTE["VARIACION_NIVEL"], alpha=0.6)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel("Extracción (lts)")
        ax.set_ylabel("ΔNivel (m)")
        ax.grid(True)
        st.pyplot(fig)

        # --- Scatter 2D con color
        st.subheader("Variación del nivel en función de extracción y lluvia")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df['EXTRACCION'], df['LLUVIA_ACUM_7D'],
                             c=df['VARIACION_NIVEL'], cmap='inferno')
        plt.colorbar(scatter, label='ΔNivel (m)', ax=ax)
        ax.set_xlabel("Extracción (lts)")
        ax.set_ylabel("Lluvia acumulada (mm)")
        ax.grid(True)
        st.pyplot(fig)

    else:
        st.info("⚠️ No hay datos cargados todavía. Subí un CSV inicial desde la barra lateral.")
