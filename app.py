# app.py — Streamlit × Wit.ai × CSV-only (list-movie, movie-info, movie-with-attribute)

import os, re, json, urllib.parse, requests, ast
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, Tuple

# =========================
# Config / Secrets
# =========================
def get_secret(k: str, default: str = "") -> str:
    try:
        return st.secrets[k]
    except Exception:
        return os.getenv(k, default)

WIT_SERVER_TOKEN = get_secret("WIT_SERVER_TOKEN", "")
API_VERSION      = get_secret("WIT_API_VERSION", "20240901")

# CSV path can be overridden via secrets if needed
CSV_PATH         = get_secret("CSV_PATH", "cleaned_movies.csv")

DEFAULT_FALLBACK = 'Tell me what you’d like to watch (e.g., "action movies from 2018").'

# Only add Authorization header if we actually have a token
HEAD = {"Authorization": f"Bearer {WIT_SERVER_TOKEN}"} if WIT_SERVER_TOKEN else {}

# =========================
# UI
# =========================
st.set_page_config(page_title="Wit.ai Movie Bot", page_icon="🎬", layout="wide")
st.title("🎬 Wit.ai Movie Bot (CSV-only)")

if not WIT_SERVER_TOKEN:
    st.warning("No Wit.ai token found in Streamlit secrets (WIT_SERVER_TOKEN). "
               "Pure year/regex routing will still work, but entity extraction from Wit is disabled.")

# Small debug panel
with st.expander("🔧 Debug"):
    st.write({
        "CSV_PATH": CSV_PATH,
        "WIT_API_VERSION": API_VERSION,
        "WIT_TOKEN_SET": bool(WIT_SERVER_TOKEN)
    })
    try:
        df_preview = pd.read_csv(CSV_PATH, nrows=3)
        st.write({"csv_loaded_rows_preview": len(df_preview)})
        st.dataframe(df_preview)
    except Exception as e:
        st.write(f"CSV load error: {e}")

# =========================
# Wit helpers
# =========================
def wit_message(text: str) -> Dict[str, Any]:
    """Query Wit.ai; if token is missing, return an empty payload so routing
    can still operate using regex/year parsing."""
    if not WIT_SERVER_TOKEN:
        return {"text": text, "intents": [], "entities": {}}
    r = requests.get(
        "https://api.wit.ai/message",
        params={"q": text, "v": API_VERSION},
        headers=HEAD,
        timeout=15
    )
    r.raise_for_status()
    return r.json()

def ent(data: Dict[str, Any], key: str) -> Optional[str]:
    arr = (data.get("entities") or {}).get(key, [])
    if not arr:
        return None
    v = arr[0].get("value")
    if isinstance(v, dict) and "value" in v:
        v = v["value"]
    return str(v) if v is not None else None

def first_year(text: str) -> Optional[str]:
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None

def extract_years(data: Dict[str, Any], text: str) -> Tuple[Optional[str], Optional[str]]:
    """Prefer custom year:year; else regex; ignore wit$datetime quirks."""
    y = None
    arr = (data.get("entities") or {}).get("year:year") or (data.get("entities") or {}).get("year") or []
    if arr:
        v = str(arr[0].get("value") or "")
        if re.fullmatch(r"(?:19|20)\d{2}", v):
            y = v
    if not y:
        y = first_year(text)
    return (y, y) if y else (None, None)

def is_pure_year(text: str) -> Optional[str]:
    s = text.strip()
    return s if re.fullmatch(r"(?:19|20)\d{2}", s) else None

def ent_any(data: Dict[str, Any], *names) -> Optional[str]:
    """Return first entity value where key equals any name OR base part before ':' equals any name."""
    ents = (data.get("entities") or {})
    for k in list(ents.keys()):
        base = k.split(":")[0]
        if k in names or base in names:
            arr = ents.get(k) or []
            if arr:
                v = arr[0].get("value")
                if isinstance(v, dict):
                    v = v.get("value")
                if v is not None:
                    return str(v)
    return None

def extract_years_any(data: Dict[str, Any], text: str) -> Tuple[Optional[str], Optional[str]]:
    # 1) entity first
    v = ent_any(data, "year")
    if v and re.fullmatch(r"(?:19|20)\d{2}", str(v)):
        return str(v), str(v)
    # 2) raw text fallback (handles pure "2021")
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m:
        y = m.group(0)
        return y, y
    return None, None

def topn_from_text(text: str) -> Optional[str]:
    m = re.search(r"\btop\s+(\d+)\b", text, re.I)
    return m.group(1) if m else None

# =========================
# CSV-only helpers / callers
# =========================
def _to_listish(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    s = str(x).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(t) for t in v]
    except Exception:
        pass
    return [p.strip() for p in re.split(r"[|;,]", s) if p.strip()]

def _pick_year_col(df: pd.DataFrame):
    for c in ["release_year", "year", "startYear", "title_year", "first_release_year"]:
        if c in df.columns:
            return c
    return None

def _load_df():
    df = pd.read_csv(CSV_PATH)
    # normalize list-like columns if present
    for c in ["genres", "cast", "director"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_listish)
    return df

def call_list_movie(params: Dict[str, str]):
    df = _load_df()

    # Year filter (supports year / year_start / year_end / release_year)
    ycol = _pick_year_col(df)
    y_start = params.get("year") or params.get("year_start") or params.get("release_year")
    y_end   = params.get("year_end") or y_start
    if ycol and y_start:
        try:
            ys = int(str(y_start)); ye = int(str(y_end)) if y_end else ys
            yvals = pd.to_numeric(df[ycol], errors="coerce")
            df = df[yvals.between(ys, ye)]
        except Exception:
            pass

    # Genre filter (exact match to one of the list values)
    if "genre" in params and "genres" in df.columns and params["genre"]:
        g = str(params["genre"]).lower()
        df = df[df["genres"].apply(lambda L: any(g == str(x).lower() for x in L))]

    # Actor filter (substring in any cast value)
    if "actor" in params and "cast" in df.columns and params["actor"]:
        a = str(params["actor"]).lower()
        df = df[df["cast"].apply(lambda L: any(a in str(x).lower() for x in L))]

    # Director filter (substring in any director value)
    if "director" in params and "director" in df.columns and params["director"]:
        d = str(params["director"]).lower()
        df = df[df["director"].apply(lambda L: any(d in str(x).lower() for x in L))]

    # "rating" used as Top-N if present
    if "rating" in params and params["rating"]:
        try:
            n = int(params["rating"])
        except Exception:
            n = None
        if n and n > 0:
            rcol = next((c for c in ["rating", "vote_average", "imdb_score"] if c in df.columns), None)
            df = (df.sort_values(rcol, ascending=False) if rcol else df).head(n)

    # Output: list of titles
    if "title" in df.columns:
        return df["title"].astype(str).tolist()
    # fallback to first column if no title col
    return df.iloc[:, 0].astype(str).tolist()

def call_movie_info(title: str):
    df = _load_df()
    if "title" not in df.columns:
        return {}
    hit = df[df["title"].str.lower() == str(title).lower()]
    if hit.empty:
        return {}
    row = hit.iloc[0]
    ycol = _pick_year_col(df)
    return {
        str(row["title"]): {
            "year_start": int(row[ycol]) if ycol and pd.notna(row[ycol]) else None,
            "directors": _to_listish(row["director"]) if "director" in df.columns else [],
            "actors": _to_listish(row["cast"]) if "cast" in df.columns else [],
            "genres": _to_listish(row["genres"]) if "genres" in df.columns else [],
            "rating": row["rating"] if "rating" in df.columns and pd.notna(row["rating"]) else None,
        }
    }

def call_movie_with_attribute(params: Dict[str, str]):
    """
    Minimal CSV version:
    - If 'movie_attribute' matches an existing column, return titles where that column is non-empty.
    - Also respects optional year filter.
    """
    df = _load_df()
    ycol = _pick_year_col(df)
    y_start = params.get("year") or params.get("year_start") or params.get("release_year")
    y_end   = params.get("year_end") or y_start
    if ycol and y_start:
        yvals = pd.to_numeric(df[ycol], errors="coerce")
        ys = int(str(y_start)); ye = int(str(y_end)) if y_end else ys
        df = df[yvals.between(ys, ye)]

    attr = str(params.get("movie_attribute") or "").strip()
    if attr and attr in df.columns:
        s = df[attr]
        if s.dtype == "O":
            df = df[s.fillna("").astype(str).str.strip() != ""]
        else:
            df = df[s.notna()]
    # return titles
    if "title" in df.columns:
        return df["title"].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()

# =========================
# Formatters (kept above Router for linters)
# =========================
def format_list(response_json, prefix="Recommended movies are:"):
    if not response_json:
        return "No movies found"

    def extract_title(item):
        if isinstance(item, (list, tuple)) and item:
            return str(item[0])
        if isinstance(item, dict):
            for k in ["title", "movie_title", "name"]:
                if k in item and item[k]:
                    return str(item[k])
            for k, v in item.items():
                if "title" in str(k).lower():
                    return str(v)
            return None
        if isinstance(item, str):
            return item
        return None

    lines = [prefix]
    for i, item in enumerate(response_json, 1):
        t = extract_title(item)
        if t:
            lines.append(f"{i}. {t}")
    return "\n".join(lines) if len(lines) > 1 else "No movies found"

def pretty_title_info(response_json, want: str):
    if not response_json:
        return "MovieTitleNotFound"
    out = []
    for title, payload in response_json.items():
        out.append(f"Movie title: {title}")
        val = payload.get(want)
        if want == "year_start":
            out.append(f"Year: {val}" if val else "Year not found")
        elif want in ("directors", "actors", "genres"):
            if isinstance(val, list) and val:
                label = "Director(s)" if want == "directors" else ("Actor(s)" if want == "actors" else "Genre(s)")
                out.append(f"{label}:")
                out.extend([f"- {x}" for x in val])
            else:
                out.append(f"{'Directors' if want=='directors' else 'Actors' if want=='actors' else 'Genres'} not found")
        elif want == "rating":
            out.append(f"Rating: {val}" if val is not None else "Rating not found")
    return "\n".join(out)

# =========================
# Router
# =========================
def route(text: str, data: Dict[str, Any]) -> str:
    # Intent (may be empty/low)
    intent = (data.get("intents") or [{}])[0].get("name")
    conf   = float((data.get("intents") or [{}])[0].get("confidence") or 0.0)

    # Entities (work even if no intent)
    director = ent_any(data, "director")
    actor    = ent_any(data, "actor")
    genre    = ent_any(data, "genre")
    title    = ent_any(data, "movie_title")
    attr_val = ent_any(data, "movie_attribute")
    y1, y2   = extract_years_any(data, text)
    ytxt     = is_pure_year(text)
    topn     = topn_from_text(text)

    # Hard override: pure-year BEFORE any fallback
    if ytxt and not any([director, actor, genre, title, attr_val]):
        p = {"year": ytxt, "year_start": ytxt, "year_end": ytxt, "release_year": ytxt}
        res = call_list_movie(p)
        return format_list(res)

    # No/low intent + no entities? default
    have_entities = any([director, actor, genre, title, attr_val, y1, y2, topn, ytxt])
    if (not intent or conf < 0.5) and not have_entities:
        return DEFAULT_FALLBACK

    # No/low intent → decide purely from entities
    if not intent or conf < 0.5:
        low = text.lower()
        if title and "director" in low: intent = "get_director_by_movie_title"
        elif title and ("actor" in low or "cast" in low): intent = "get_actor_by_movie_title"
        elif title and "year" in low: intent = "get_year_by_movie_title"
        elif title and "genre" in low: intent = "get_genre_by_movie_title"
        elif title and "rating" in low: intent = "get_rating_by_movie_title"
        else:
            if (y1 or ytxt) and not any([director, actor, genre, title, attr_val]):
                y = y1 or ytxt
                p = {"year": y, "year_start": y, "year_end": y, "release_year": y}
                res = call_list_movie(p); return format_list(res)
            if director and not any([actor, genre, y1, title, attr_val]):
                res = call_list_movie({"director": director}); return format_list(res)
            if actor and not any([director, genre, y1, title, attr_val]):
                res = call_list_movie({"actor": actor}); return format_list(res)
            if genre and not any([director, actor, y1, title, attr_val]):
                res = call_list_movie({"genre": genre}); return format_list(res)
            if any([director, actor, genre, y1, ytxt]):
                y = y1 or ytxt
                p = {}
                if director: p["director"] = director
                if actor:    p["actor"]    = actor
                if genre:    p["genre"]    = genre
                if y:        p["year"] = p["year_start"] = y
                res = call_list_movie(p); return format_list(res)

    # Intent handlers (still supported)
    if intent == "movie_match_director" and director:
        res = call_list_movie({"director": director}); return format_list(res)
    if intent == "movie_match_actor" and actor:
        res = call_list_movie({"actor": actor}); return format_list(res)
    if intent == "movie_match_genre" and genre:
        res = call_list_movie({"genre": genre}); return format_list(res)
    if intent == "movie_match_year" and (y1 or y2 or ytxt):
        y = y1 or ytxt
        p = {}
        if y:  p["year"] = p["year_start"] = y
        if y2: p["year_end"] = y2
        res = call_list_movie(p); return format_list(res)
    if intent == "movie_match_rating":
        p = {}
        if topn:     p["rating"]   = topn
        if genre:    p["genre"]    = genre
        if director: p["director"] = director
        if actor:    p["actor"]    = actor
        if (y1 or ytxt):
            y = y1 or ytxt
            p["year"] = p["year_start"] = y
        res = call_list_movie(p); return format_list(res)
    if intent == "movie_match_several_criteria":
        p = {}
        if director: p["director"] = director
        if actor:    p["actor"]    = actor
        if genre:    p["genre"]    = genre
        if (y1 or ytxt):
            y = y1 or ytxt
            p["year"] = p["year_start"] = y
        res = call_list_movie(p); return format_list(res)

    if intent == "get_director_by_movie_title" and title:
        info = call_movie_info(title); return pretty_title_info(info, want="directors")
    if intent == "get_actor_by_movie_title" and title:
        info = call_movie_info(title); return pretty_title_info(info, want="actors")
    if intent == "get_year_by_movie_title" and title:
        info = call_movie_info(title); return pretty_title_info(info, want="year_start")
    if intent == "get_genre_by_movie_title" and title:
        info = call_movie_info(title); return pretty_title_info(info, want="genres")
    if intent == "get_rating_by_movie_title" and title:
        info = call_movie_info(title); return pretty_title_info(info, want="rating")
    if intent == "get_movie_attributes" and attr_val:
        p = {"movie_attribute": attr_val}
        if (y1 or ytxt):
            y = y1 or ytxt
            p["year"] = p["year_start"] = y
        res = call_movie_with_attribute(p); return format_list(res)

    # Last resort
    if (y1 or ytxt):
        y = y1 or ytxt
        p = {"year": y, "year_start": y, "year_end": y2 or y, "release_year": y}
        res = call_list_movie(p); return format_list(res)

    return DEFAULT_FALLBACK

# =========================
# Chat Loop
# =========================
for m in st.session_state.get("history", []):
    with st.chat_message(m["role"]):
        st.markdown(m["text"])

prompt = st.chat_input("Ask me about movies…")
if prompt:
    st.session_state.setdefault("history", []).append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        data = wit_message(prompt)
        answer = route(prompt, data)
    except Exception as e:
        answer = f"Error: {e}"

    st.session_state["history"].append({"role": "assistant", "text": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
