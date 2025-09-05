# app.py — Streamlit × Wit.ai × cleaned_movies.csv
import os, re, json, difflib, requests
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Tuple, Optional

# ---------- page setup (hide sidebar) ----------
st.set_page_config(page_title="Wit.ai Netflix Recommender", page_icon="🎬",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
/* hide the entire left sidebar */
section[data-testid="stSidebar"] { display: none !important; }
/* optional: hide the hamburger that toggles the sidebar */
// button[kind="header"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ---------- config ----------
def get_secret(k, default=""):
    try: return st.secrets[k]
    except Exception: return os.getenv(k, default)

WIT_SERVER_TOKEN = get_secret("WIT_SERVER_TOKEN", "PASTE_YOUR_WIT_SERVER_ACCESS_TOKEN_HERE")
API_VERSION = "20240901"
CSV_PATH = "cleaned_movies.csv"
CONF_INTENT_MIN = 0.50
MAX_RESULTS_DEFAULT = 10

if not WIT_SERVER_TOKEN or "PASTE_" in WIT_SERVER_TOKEN:
    st.warning("Add your **WIT_SERVER_TOKEN** in Streamlit Secrets or as an env var.")

# ---------- data ----------
@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["title","genres","director","cast","overview","description",
              "vote_average","rating","vote_count","popularity","release_year","year"]:
        if c in df.columns and df[c].dtype == "O":
            df[c] = df[c].fillna("")
    if "description" not in df.columns and "overview" in df.columns:
        df["description"] = df["overview"].fillna("")
    if "release_year" not in df.columns and "year" in df.columns:
        df["release_year"] = df["year"]
    return df

df = load_df(CSV_PATH)

def col(*names): 
    for n in names:
        if n in df.columns: return n
    return None

COL_TITLE = col("title")
COL_GENRE = col("genres")
COL_DIR   = col("director")
COL_CAST  = col("cast")
COL_YEAR  = col("release_year","year")
COL_DESC  = col("description","overview")
COL_RATE  = col("vote_average","rating")
COL_VOTE  = col("vote_count")
COL_POP   = col("popularity")

def s_contains(series: pd.Series, needle: str) -> pd.Series:
    return series.str.lower().str.contains(re.escape(needle.lower()), na=False)

def rank_df(dd: pd.DataFrame) -> pd.DataFrame:
    order = [c for c in [COL_RATE, COL_VOTE, COL_POP] if c]
    return dd.sort_values(by=order, ascending=[False]*len(order)) if order else dd

def tidy_table(dd: pd.DataFrame, k: int) -> pd.DataFrame:
    # choose user-friendly columns and drop any id-like column
    cols = [c for c in [COL_TITLE, COL_YEAR, COL_GENRE, COL_RATE] if c]
    table = dd[cols].head(k).copy()
    # hide index and any column literally named 'id'
    if "id" in table.columns: table = table.drop(columns=["id"])
    return table.reset_index(drop=True)

# ---------- wit ----------
HEAD = {"Authorization": f"Bearer {WIT_SERVER_TOKEN}"}
def wit_message(text: str) -> Dict[str, Any]:
    r = requests.get("https://api.wit.ai/message",
                     params={"q": text, "v": API_VERSION},
                     headers=HEAD, timeout=15)
    r.raise_for_status()
    return r.json()

def top_intent(d: Dict[str, Any]): 
    arr = d.get("intents") or []
    return (arr[0].get("name"), float(arr[0].get("confidence") or 0.0)) if arr else (None, 0.0)

def ent(d: Dict[str, Any], key: str) -> Optional[str]:
    arr = (d.get("entities") or {}).get(key, []) or []
    if not arr: return None
    v = arr[0].get("value")
    return v["value"] if isinstance(v, dict) and "value" in v else (str(v) if v is not None else None)

def years_from_datetime(d: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    ents = d.get("entities") or {}
    dt = ents.get("wit$datetime:datetime") or ents.get("wit$datetime") or []
    if not dt: return None, None
    v = dt[0].get("value")
    if isinstance(v, dict) and "from" in v and "to" in v:
        try: return int(v["from"]["value"][:4]), int(v["to"]["value"][:4])
        except: return None, None
    if isinstance(v, str) and re.match(r"^\d{4}", v): 
        y = int(v[:4]); return y, y
    return None, None

def year_from_text(text: str) -> Optional[int]:
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return int(m.group(1)) if m else None

def topn_from_text(text: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)\b", text, re.I)
    return int(m.group(1)) if m else None

# ---------- router: returns (header_text, table_df) ----------
def route(text: str, data: Dict[str, Any], k=MAX_RESULTS_DEFAULT, conf_min=CONF_INTENT_MIN):
    intent, conf = top_intent(data)

    # entities
    director = ent(data, "director:director") or ent(data, "director")
    actor    = ent(data, "actor:actor") or ent(data, "actor")
    genre    = ent(data, "genre:genre") or ent(data, "genre")
    title    = ent(data, "movie_title:movie_title") or ent(data, "movie_title")
    attr     = ent(data, "movie_attribute:movie_attribute") or ent(data, "movie_attribute")
    y1, y2   = years_from_datetime(data)
    year_guess = year_from_text(text)
    topn_guess = topn_from_text(text)

    # friendly headers helper
    def header(prefix: str, **kw):
        bits = [prefix]
        if kw.get("year"):  bits.append(f"in {kw['year']}")
        if kw.get("genre"): bits.append(f"for {kw['genre']}")
        if kw.get("director"): bits.append(f"by {kw['director']}")
        if kw.get("actor"): bits.append(f"with {kw['actor']}")
        return " ".join(bits) + "."

    if intent is None or conf < conf_min:
        return ("Tell me what you’d like to watch (e.g., “action movies from 2018”).", pd.DataFrame())

    # small-talk
    if intent in {"greet","goodbye","affirm","deny","mood_great","mood_unhappy","bot_challenge"}:
        msg = {
            "greet":"Hello! What are you in the mood to watch?",
            "goodbye":"Bye! Happy watching.",
            "affirm":"Great! Tell me more.",
            "deny":"No worries—what would you like instead?",
            "mood_great":"Awesome 😄 Want a feel-good pick?",
            "mood_unhappy":"Sorry to hear that. Want something uplifting?",
            "bot_challenge":"I’m an assistant powered by Wit.ai + Streamlit."
        }[intent]
        return (msg, pd.DataFrame())

    # queries
    if intent == "movie_match_year":
        yr = y1 or y2 or year_guess
        if yr and COL_YEAR:
            dd = df[df[COL_YEAR].astype(str).str.contains(str(yr), na=False)]
            dd = rank_df(dd)
            table = tidy_table(dd, topn_guess or k)
            return (f"Here are {len(table)} movies released in {yr}.", table)

    if intent == "movie_match_genre" and genre:
        dd = rank_df(df[s_contains(df[COL_GENRE], genre)]) if COL_GENRE else df.iloc[0:0]
        table = tidy_table(dd, k)
        return (header(f"Here are {len(table)} {genre} movies"), table)

    if intent == "movie_match_director" and director:
        dd = rank_df(df[s_contains(df[COL_DIR], director)]) if COL_DIR else df.iloc[0:0]
        table = tidy_table(dd, k)
        return (header(f"Here are {len(table)} movies", director=director), table)

    if intent == "movie_match_actor" and actor:
        dd = rank_df(df[s_contains(df[COL_CAST], actor)]) if COL_CAST else df.iloc[0:0]
        table = tidy_table(dd, k)
        return (header(f"Here are {len(table)} movies", actor=actor), table)

    if intent == "movie_match_rating":
        yr = y1 or y2 or year_guess
        dd = df
        if yr and COL_YEAR:
            dd = dd[dd[COL_YEAR].astype(str).str.contains(str(yr), na=False)]
        dd = rank_df(dd)
        n = topn_guess or 10
        table = tidy_table(dd, n)
        return (f"Top {len(table)} rated movies" + (f" in {yr}" if yr else "") + ".", table)

    if intent == "movie_match_several_criteria":
        m = pd.Series([True]*len(df))
        if genre and COL_GENRE:  m &= s_contains(df[COL_GENRE], genre)
        if actor and COL_CAST:   m &= s_contains(df[COL_CAST], actor)
        if director and COL_DIR: m &= s_contains(df[COL_DIR], director)
        if (y1 or y2) and COL_YEAR:
            y1_ = y1 or 1800; y2_ = y2 or 2100
            years = pd.to_numeric(df[COL_YEAR], errors="coerce").fillna(-1).astype(int)
            m &= years.between(y1_, y2_)
        dd = rank_df(df[m])
        table = tidy_table(dd, k)
        return (header(f"Here are {len(table)} movies", genre=genre, actor=actor, director=director,
                       year=f"{y1}-{y2}" if (y1 and y2 and y1!=y2) else (y1 or y2)), table)

    if intent.startswith("get_") and title:
        # simple fact lookups
        from difflib import get_close_matches
        choices = df[COL_TITLE].astype(str).tolist() if COL_TITLE else []
        found = get_close_matches(title, choices, n=1, cutoff=0.6)
        if not found: return (f"Couldn't find a title like '{title}'.", pd.DataFrame())
        row = df[df[COL_TITLE].str.lower()==found[0].lower()].iloc[0]
        if intent == "get_director_by_movie_title":
            val = row.get(COL_DIR, "not available"); return (f"Director(s) of {found[0]}: {val}", pd.DataFrame())
        if intent == "get_actor_by_movie_title":
            val = row.get(COL_CAST, "not available"); return (f"Actors in {found[0]}: {val}", pd.DataFrame())
        if intent == "get_year_by_movie_title":
            val = row.get(COL_YEAR, "not available"); return (f"Year of {found[0]}: {val}", pd.DataFrame())
        if intent == "get_genre_by_movie_title":
            val = row.get(COL_GENRE, "not available"); return (f"Genre(s) of {found[0]}: {val}", pd.DataFrame())
        if intent == "get_rating_by_movie_title":
            val = row.get(COL_RATE, "not available"); return (f"Rating of {found[0]}: {val}", pd.DataFrame())

    if intent == "get_movie_attributes" and attr := (ent(data,"movie_attribute:movie_attribute") or ent(data,"movie_attribute")):
        dd = df.copy()
        if genre and COL_GENRE: dd = dd[s_contains(dd[COL_GENRE], genre)]
        if (y1 or y2) and COL_YEAR:
            y1_ = y1 or 1800; y2_ = y2 or 2100
            years = pd.to_numeric(dd[COL_YEAR], errors="coerce").fillna(-1).astype(int)
            dd = dd[years.between(y1_, y2_)]
        if COL_DESC:
            score = dd[COL_DESC].str.lower().str.count(re.escape(attr.lower()))
            dd = dd.assign(_attr=score).sort_values(by=["_attr", COL_POP or COL_RATE or COL_VOTE], ascending=[False, False])
        table = tidy_table(dd, k)
        return (header(f"Here are {len(table)} movies with", year=(y1 or y2), genre=genre)[:-1] + f' “{attr}”.', table)

    return ("I couldn’t parse that. Try: “tell me action movies”, “movies directed by Quentin Tarantino”, or “top 5 rated movies in 2018”.", pd.DataFrame())

# ---------- UI ----------
st.title("🎬 Netflix Recommender")
st.caption("Ask anything: “movies released in 2025”, “comedy movies with Tom Hanks”, “top 5 rated movies in 2019”.")

# chat history (optional)
st.session_state.setdefault("history", [])

for m in st.session_state["history"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "table" in m and not m["table"].empty:
            st.dataframe(m["table"], use_container_width=True)

text = st.chat_input("What are you in the mood for?")
if text:
    st.session_state["history"].append({"role":"user","content":text})
    with st.chat_message("user"): st.markdown(text)

    try:
        data = wit_message(text)
        msg, table = route(text, data)
    except Exception as e:
        msg, table = (f"Error: {e}", pd.DataFrame())

    st.session_state["history"].append({"role":"assistant","content":msg,"table":table})
    with st.chat_message("assistant"):
        st.markdown(msg)
        if not table.empty:
            st.dataframe(table, use_container_width=True)
