# app.py — Streamlit × Wit.ai × endpoints (list-movie, movie-info, movie-with-attribute)
import os, re, json, urllib.parse, requests
import streamlit as st
from typing import Dict, Any, Optional, Tuple

# ---------- Config ----------
WIT_SERVER_TOKEN = os.getenv("WIT_SERVER_TOKEN", "PASTE_YOUR_WIT_SERVER_ACCESS_TOKEN_HERE")
API_VERSION      = os.getenv("WIT_API_VERSION", "20240901")

ENDPOINT_DATABASE_PATH = os.getenv("MOVIE_API_BASE", "http://localhost:8000")
ENDPOINT_GET_MOVIE = "/list-movie"
ENDPOINT_GET_MOVIE_INFO = "/movie-info"
ENDPOINT_GET_MOVIE_WITH_ATTRIBUTES = "/movie-with-attribute"
DEFAULT_FALLBACK = 'Tell me what you’d like to watch (e.g., "action movies from 2018").'

HEAD = {"Authorization": f"Bearer {WIT_SERVER_TOKEN}"}

# ---------- UI ----------
st.set_page_config(page_title="Wit.ai Movie Bot", page_icon="🎬", layout="wide")
st.title("🎬 Wit.ai Movie Bot (Rasa-style actions)")
if not WIT_SERVER_TOKEN or "PASTE_" in WIT_SERVER_TOKEN:
    st.warning("Set WIT_SERVER_TOKEN env var.")

# ---------- Wit helpers ----------
def wit_message(text: str) -> Dict[str, Any]:
    r = requests.get("https://api.wit.ai/message",
                     params={"q": text, "v": API_VERSION},
                     headers=HEAD, timeout=15)
    r.raise_for_status()
    return r.json()

def ent(data: Dict[str, Any], key: str) -> Optional[str]:
    arr = (data.get("entities") or {}).get(key, [])
    if not arr: return None
    v = arr[0].get("value")
    if isinstance(v, dict) and "value" in v: v = v["value"]
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
        if re.fullmatch(r"(?:19|20)\d{2}", v): y = v
    if not y:
        y = first_year(text)
    return (y, y) if y else (None, None)

def is_pure_year(text: str) -> Optional[str]:
    s = text.strip()
    return s if re.fullmatch(r"(?:19|20)\d{2}", s) else None

def ent_any(data: Dict[str, Any], *names) -> Optional[str]:
    """Return first entity value where key equals any name OR base part before ':' equals any name."""
    ents = (data.get("entities") or {})
    keys = list(ents.keys())
    for k in keys:
        base = k.split(":")[0]
        if k in names or base in names:
            arr = ents.get(k) or []
            if arr:
                v = arr[0].get("value")
                if isinstance(v, dict): v = v.get("value")
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

# ---------- Endpoint callers ----------
def call_list_movie(params: Dict[str, str]):
    url = ENDPOINT_DATABASE_PATH + ENDPOINT_GET_MOVIE
    if params:
        qs = "&".join(f"{k}={urllib.parse.quote(v)}" for k, v in params.items() if v)
        url = f"{url}?{qs}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def call_movie_info(title: str):
    url = ENDPOINT_DATABASE_PATH + ENDPOINT_GET_MOVIE_INFO + "?movie_title=" + urllib.parse.quote(title)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def call_movie_with_attribute(params: Dict[str, str]):
    url = ENDPOINT_DATABASE_PATH + ENDPOINT_GET_MOVIE_WITH_ATTRIBUTES
    if params:
        qs = "&".join(f"{k}={urllib.parse.quote(v)}" for k, v in params.items() if v)
        url = f"{url}?{qs}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------- Formatters ----------
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

# ---------- Router ----------
def route(text: str, data: Dict[str, Any]) -> str:
    # ---------- INTENT (may be empty/low) ----------
    intent = (data.get("intents") or [{}])[0].get("name")
    conf   = float((data.get("intents") or [{}])[0].get("confidence") or 0.0)

    # ---------- ENTITIES (work even if no intent) ----------
    director = ent_any(data, "director")
    actor    = ent_any(data, "actor")
    genre    = ent_any(data, "genre")
    title    = ent_any(data, "movie_title")
    attr_val = ent_any(data, "movie_attribute")
    y1, y2   = extract_years_any(data, text)

    # NEW: detect a typed year like "2020" up-front
    ytxt = is_pure_year(text)

    topn = topn_from_text(text)

    # ---------- HARD OVERRIDE: pure-year BEFORE any fallback ----------
    if ytxt and not any([director, actor, genre, title, attr_val]):
        p = {"year": ytxt, "year_start": ytxt, "year_end": ytxt, "release_year": ytxt}
        res = call_list_movie(p)
        return format_list(res)

    # ---------- NO/LOW INTENT + NO ENTITIES? show default ----------
    # Count ytxt as an "entity" too, so we don't fallback on "2020"
    have_entities = any([director, actor, genre, title, attr_val, y1, y2, topn, ytxt])
    if (not intent or conf < 0.5) and not have_entities:
        return DEFAULT_FALLBACK

    # ---------- NO/LOW INTENT? Decide purely from entities ----------
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

    # ---------- Normal intent handlers (unchanged) ----------
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

    # ---------- Last resort ----------
    if (y1 or ytxt):
        y = y1 or ytxt
        p = {"year": y, "year_start": y, "year_end": y2 or y, "release_year": y}
        res = call_list_movie(p); return format_list(res)

    return DEFAULT_FALLBACK

# ---------- Chat ----------
for m in st.session_state.get("history", []):
    with st.chat_message(m["role"]):
        st.markdown(m["text"])

prompt = st.chat_input("Ask me about movies…")
if prompt:
    st.session_state.setdefault("history", []).append({"role":"user","text":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    try:
        data = wit_message(prompt)
        answer = route(prompt, data)
    except Exception as e:
        answer = f"Error: {e}"

    st.session_state["history"].append({"role":"assistant","text":answer})
    with st.chat_message("assistant"): st.markdown(answer)
