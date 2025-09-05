import os, re, json, difflib, requests
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Tuple, Optional

# ----------------------------- CONFIG -----------------------------
st.set_page_config(page_title="Wit.ai × Netflix Recommender", page_icon="🎬", layout="wide")

def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]  # from .streamlit/secrets.toml
    except Exception:
        return os.getenv(key, default)

WIT_SERVER_TOKEN = get_secret("WIT_SERVER_TOKEN", "PASTE_YOUR_WIT_SERVER_ACCESS_TOKEN_HERE")
API_VERSION = "20240901"
CSV_PATH_DEFAULT = "cleaned_movies.csv"
MAX_RESULTS_DEFAULT = 10
CONF_INTENT_MIN = 0.50
# -----------------------------------------------------------------

# ----------------------------- UI (SIDEBAR) -----------------------
with st.sidebar:
    st.header("Settings")
    token = st.text_input("Wit Server Access Token", value=WIT_SERVER_TOKEN, type="password")
    csv_choice = st.text_input("CSV path", value=CSV_PATH_DEFAULT, help="Path to your cleaned_movies.csv")
    uploaded = st.file_uploader("...or upload cleaned_movies.csv", type=["csv"])
    max_results = st.slider("Max results to show", 3, 30, MAX_RESULTS_DEFAULT, 1)
    conf_min = st.slider("Min intent confidence", 0.0, 1.0, CONF_INTENT_MIN, 0.05)
    show_debug = st.checkbox("Show raw Wit JSON in debug expander", value=False)

if not token or "PASTE_" in token:
    st.warning("Add your **Wit Server Access Token** in the sidebar (or in `.streamlit/secrets.toml`).")

# ----------------------------- DATA LOADING -----------------------
@st.cache_data(show_spinner=False)
def load_df(file: Optional[str], uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(file)
    # normalize common columns
    for c in ["title","type","genres","director","cast","language","overview","description",
              "vote_average","vote_count","popularity","release_year","year","runtime","maturity_rating"]:
        if c in df.columns:
            if df[c].dtype == "O":
                df[c] = df[c].fillna("")
    if "description" not in df.columns and "overview" in df.columns:
        df["description"] = df["overview"].fillna("")
    if "release_year" not in df.columns and "year" in df.columns:
        df["release_year"] = df["year"]
    return df

try:
    df = load_df(csv_choice, uploaded)
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

def pick_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    for n in names:
        if n in df.columns: return n
    return None

COL_TITLE  = pick_col(df, "title")
COL_GENRE  = pick_col(df, "genres")
COL_DIR    = pick_col(df, "director")
COL_CAST   = pick_col(df, "cast")
COL_YEAR   = pick_col(df, "release_year","year")
COL_DESC   = pick_col(df, "description","overview")
COL_RATE   = pick_col(df, "vote_average","rating")
COL_VOTES  = pick_col(df, "vote_count")
COL_POP    = pick_col(df, "popularity")

# ----------------------------- HELPERS ----------------------------
def s_contains(series: pd.Series, needle: str) -> pd.Series:
    return series.str.lower().str.contains(re.escape(needle.lower()), na=False)

def rank_df(dd: pd.DataFrame) -> pd.DataFrame:
    order = []
    if COL_RATE:  order.append(COL_RATE)
    if COL_VOTES: order.append(COL_VOTES)
    if COL_POP:   order.append(COL_POP)
    if not order: return dd
    return dd.sort_values(by=order, ascending=[False]*len(order))

def format_rows(dd: pd.DataFrame, k: int) -> Tuple[str, List[Dict[str, Any]], pd.DataFrame]:
    cols = [c for c in [COL_TITLE,COL_YEAR,COL_GENRE,COL_RATE] if c]
    show = dd[cols].head(k).copy()
    items = show.to_dict(orient="records")
    lines = []
    for _, r in show.iterrows():
        bits = [str(r.get(COL_TITLE,""))]
        if COL_YEAR: bits.append(str(r.get(COL_YEAR,"")).split(".")[0])
        if COL_GENRE: bits.append(str(r.get(COL_GENRE,"")))
        if COL_RATE: bits.append(f"⭐ {r.get(COL_RATE)}")
        lines.append(" — ".join([b for b in bits if b]))
    return ("\n".join(lines) if lines else "No matches."), items, show

def fuzzy_title(df: pd.DataFrame, title: str) -> Optional[str]:
    if not COL_TITLE: return None
    mask = s_contains(df[COL_TITLE], title)
    if mask.any():
        return df[mask].iloc[0][COL_TITLE]
    choices = df[COL_TITLE].astype(str).tolist()
    match = difflib.get_close_matches(title, choices, n=1, cutoff=0.6)
    return match[0] if match else None

def year_from_text(text: str) -> Optional[int]:
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return int(m.group(1)) if m else None

def topn_from_text(text: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)\b", text, re.I)
    return int(m.group(1)) if m else None

# ----------------------------- WIT API ----------------------------
def wit_message(text: str, server_token: str) -> Dict[str, Any]:
    head = {"Authorization": f"Bearer {server_token}"}
    r = requests.get("https://api.wit.ai/message",
                     params={"q": text, "v": API_VERSION},
                     headers=head, timeout=15)
    r.raise_for_status()
    return r.json()

def top_intent(data: Dict[str, Any]) -> Tuple[Optional[str], float]:
    intents = data.get("intents") or []
    if not intents: return None, 0.0
    i0 = intents[0]
    return i0.get("name"), float(i0.get("confidence") or 0.0)

def ent(data: Dict[str, Any], key: str) -> Optional[str]:
    arr = (data.get("entities") or {}).get(key, [])
    if arr:
        v = arr[0].get("value")
        if isinstance(v, dict) and "value" in v: v = v["value"]
        return str(v) if v is not None else None
    return None

def years_from_datetime(data: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    ents = data.get("entities") or {}
    dt = ents.get("wit$datetime:datetime") or ents.get("wit$datetime") or []
    if not dt: return None, None
    v = dt[0].get("value")
    if isinstance(v, dict) and "from" in v and "to" in v:
        try:
            y1 = int(v["from"]["value"][:4]); y2 = int(v["to"]["value"][:4])
            return y1, y2
        except Exception:
            return None, None
    if isinstance(v, str) and re.match(r"^\d{4}", v):
        return int(v[:4]), int(v[:4])
    return None, None

# ----------------------------- INTENT HANDLERS --------------------
def handle_by_director(director: str, k: int):
    if not COL_DIR: return "No 'director' column.", None
    m = s_contains(df[COL_DIR], director)
    return format_rows(rank_df(df[m]), k)

def handle_by_actor(actor: str, k: int):
    if not COL_CAST: return "No 'cast' column.", None
    m = s_contains(df[COL_CAST], actor)
    return format_rows(rank_df(df[m]), k)

def handle_by_genre(genre: str, k: int):
    if not COL_GENRE: return "No 'genres' column.", None
    m = s_contains(df[COL_GENRE], genre)
    return format_rows(rank_df(df[m]), k)

def handle_by_year(year: int, k: int):
    if not COL_YEAR: return "No 'release_year' column.", None
    m = df[COL_YEAR].astype(str).str.contains(str(year), na=False)
    return format_rows(rank_df(df[m]), k)

def handle_top_rated(year: Optional[int], topn: int):
    dd = df
    if year and COL_YEAR:
        dd = dd[dd[COL_YEAR].astype(str).str.contains(str(year), na=False)]
    return format_rows(rank_df(dd), topn or MAX_RESULTS_DEFAULT)

def handle_several(genre: Optional[str], actor: Optional[str], director: Optional[str],
                   y1: Optional[int], y2: Optional[int], k: int):
    m = pd.Series([True]*len(df))
    if genre and COL_GENRE:   m &= s_contains(df[COL_GENRE], genre)
    if actor and COL_CAST:    m &= s_contains(df[COL_CAST], actor)
    if director and COL_DIR:  m &= s_contains(df[COL_DIR], director)
    if (y1 or y2) and COL_YEAR:
        y1 = y1 or 1800; y2 = y2 or 2100
        years = pd.to_numeric(df[COL_YEAR], errors="coerce").fillna(-1).astype(int)
        m &= years.between(y1, y2)
    return format_rows(rank_df(df[m]), k)

def handle_info_by_title(title: str, field: str):
    if not COL_TITLE: return "No 'title' column.", None
    found = fuzzy_title(df, title)
    if not found:
        return f"Couldn't find a title similar to '{title}'.", None
    row = df[df[COL_TITLE].str.lower() == found.lower()].iloc[0]
    if field == "director":
        val = row.get(COL_DIR, "")
        msg = f"Director(s) of {found}: {val or 'not available'}"
    elif field == "actors":
        val = row.get(COL_CAST, "")
        msg = f"Actors in {found}: {val or 'not available'}"
    elif field == "year":
        val = row.get(COL_YEAR, "")
        msg = f"Year of {found}: {val or 'not available'}"
    elif field == "genre":
        val = row.get(COL_GENRE, "")
        msg = f"Genre(s) of {found}: {val or 'not available'}"
    elif field == "rating":
        val = row.get(COL_RATE, "")
        msg = f"Rating of {found}: {val or 'not available'}"
    else:
        msg = f"{found}"
    return msg, pd.DataFrame([{COL_TITLE: found, field: val}])

def handle_attribute(attr: str, genre: Optional[str], y1: Optional[int], y2: Optional[int], k: int):
    dd = df.copy()
    if genre and COL_GENRE:
        dd = dd[s_contains(dd[COL_GENRE], genre)]
    if (y1 or y2) and COL_YEAR:
        y1 = y1 or 1800; y2 = y2 or 2100
        years = pd.to_numeric(dd[COL_YEAR], errors="coerce").fillna(-1).astype(int)
        dd = dd[years.between(y1, y2)]
    if COL_DESC:
        score = dd[COL_DESC].str.lower().str.count(re.escape(attr.lower()))
        dd = dd.assign(__attr_score__=score).sort_values(
            by=["__attr_score__"] + [c for c in [COL_POP,COL_RATE,COL_VOTES] if c],
            ascending=[False, False, False, False]
        )
    return format_rows(dd, k)

# ----------------------------- ROUTER -----------------------------
def route_wit(text: str, data: Dict[str, Any], k: int, conf_min: float) -> Tuple[str, Optional[pd.DataFrame], Dict[str, Any]]:
    intent, conf = top_intent(data)

    # Entities (support both 'entity' and 'entity:role')
    director = ent(data, "director:director") or ent(data, "director")
    actor    = ent(data, "actor:actor") or ent(data, "actor")
    genre    = ent(data, "genre:genre") or ent(data, "genre")
    title    = ent(data, "movie_title:movie_title") or ent(data, "movie_title")
    attr     = ent(data, "movie_attribute:movie_attribute") or ent(data, "movie_attribute")

    y1, y2 = years_from_datetime(data)
    year_guess = year_from_text(text)
    topn_guess = topn_from_text(text)

    if intent is None or conf < conf_min:
        return ("I can filter by director, actor, genre, year, and rating. Try: 'tell me action movies from 2018'.", None, data)

    # Small heuristic: “directed by …” phrasing -> director intent
    if intent == "get_movie_attributes":
        m = re.search(r"directed by\s+([A-Za-z][A-Za-z .'\-]+)", text, re.I)
        if m and not director:
            director = m.group(1).strip().rstrip("?.! ")
            intent = "movie_match_director"

    # Small-talk intents
    if intent in ("greet","goodbye","affirm","deny","mood_great","mood_unhappy","bot_challenge"):
        reply_map = {
            "greet":"Hello! What are you in the mood to watch?",
            "goodbye":"Bye! Happy watching.",
            "affirm":"Great! Tell me more.",
            "deny":"No worries—what would you like instead?",
            "mood_great":"Awesome 😄 Want a feel-good pick?",
            "mood_unhappy":"Sorry to hear that. Want something uplifting?",
            "bot_challenge":"I’m an assistant powered by Wit.ai + Streamlit."
        }
        return (reply_map[intent], None, data)

    # Movie queries
    if intent == "movie_match_director" and director:
        msg, _, table = handle_by_director(director, k); return (msg, table, data)
    if intent == "movie_match_actor" and actor:
        msg, _, table = handle_by_actor(actor, k); return (msg, table, data)
    if intent == "movie_match_genre" and genre:
        msg, _, table = handle_by_genre(genre, k); return (msg, table, data)
    if intent == "movie_match_year":
        yr = y1 or y2 or year_guess
        if yr:
            msg, _, table = handle_by_year(yr, k); return (msg, table, data)

    if intent == "movie_match_rating":
        yr = y1 or y2 or year_guess
        topn = topn_guess or 10
        msg, _, table = handle_top_rated(yr, topn); return (msg, table, data)

    if intent == "movie_match_several_criteria":
        msg, _, table = handle_several(genre, actor, director, y1 or year_guess, y2, k)
        return (msg, table, data)

    if intent == "get_director_by_movie_title" and title:
        msg, table = handle_info_by_title(title, "director"); return (msg, table, data)
    if intent == "get_actor_by_movie_title" and title:
        msg, table = handle_info_by_title(title, "actors"); return (msg, table, data)
    if intent == "get_year_by_movie_title" and title:
        msg, table = handle_info_by_title(title, "year"); return (msg, table, data)
    if intent == "get_genre_by_movie_title" and title:
        msg, table = handle_info_by_title(title, "genre"); return (msg, table, data)
    if intent == "get_rating_by_movie_title" and title:
        msg, table = handle_info_by_title(title, "rating"); return (msg, table, data)

    if intent == "get_movie_attributes" and attr:
        msg, _, table = handle_attribute(attr, genre, y1, y2, k)
        return (msg, table, data)

    # Fallbacks
    if director: msg, _, table = handle_by_director(director, k); return (msg, table, data)
    if actor:    msg, _, table = handle_by_actor(actor, k); return (msg, table, data)
    if genre:    msg, _, table = handle_by_genre(genre, k); return (msg, table, data)
    if year_guess: msg, _, table = handle_by_year(year_guess, k); return (msg, table, data)

    return ("I couldn’t parse that. Try: 'tell me action movies', 'movies directed by Quentin Tarantino', or 'top 5 rated movies in 2018'.", None, data)

# ----------------------------- CHAT UI ----------------------------
st.title("🎬 Netflix Recommender (Wit.ai + Streamlit)")
st.caption("Type natural queries. I’ll parse with Wit.ai and search your CSV.")

st.session_state.setdefault("history", [])

# Show history
for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("table") is not None:
            st.dataframe(msg["table"], use_container_width=True)

# Input
user_text = st.chat_input("What are you in the mood for?")
if user_text:
    # display user msg
    st.session_state["history"].append({"role":"user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # call Wit
    try:
        data = wit_message(user_text, token)
    except requests.HTTPError as e:
        detail = getattr(e.response, "text", "") or str(e)
        reply = f"Error calling Wit.ai: {detail[:500]}"
        st.session_state["history"].append({"role":"assistant", "content": reply})
        with st.chat_message("assistant"):
            st.error(reply)
    else:
        # route and render
        reply, table, raw = route_wit(user_text, data, max_results, conf_min)
        st.session_state["history"].append({"role":"assistant", "content": reply, "table": table})
        with st.chat_message("assistant"):
            st.markdown(reply)
            if table is not None:
                st.dataframe(table, use_container_width=True)

        if show_debug:
            with st.expander("Debug: Wit raw JSON"):
                st.code(json.dumps(raw, indent=2, ensure_ascii=False), language="json")
