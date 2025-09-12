# app.py — Streamlit × Wit.ai × CSV-only, pretty table results

import os, re, json, urllib.parse, requests, ast
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, Tuple
from difflib import get_close_matches

# =========================
# Config / Secrets
# =========================
def get_secret(k: str, default: str = "") -> str:
    try:
        return st.secrets[k]
    except Exception:
        return os.getenv(k, default)

def _as_int(x, default: int) -> int:
    try:
        return int(str(x))
    except Exception:
        return default

WIT_SERVER_TOKEN = get_secret("WIT_SERVER_TOKEN", "")
API_VERSION      = get_secret("WIT_API_VERSION", "20240901")
CSV_PATH         = get_secret("CSV_PATH", "cleaned_movies.csv")
RESULTS_LIMIT    = _as_int(get_secret("RESULTS_LIMIT", "20"), 20)

DEFAULT_FALLBACK = 'Tell me what you’d like to watch (e.g., "action movies from 2018").'
ACTORISH    = re.compile(r"\b(actor|cast|star|starring|featured|featuring|appears|acted|performer|participant|play(?:ed|ing)?)\b", re.I)
DIRECTORISH = re.compile(r"\b(direct(?:or|ed|ing)|filmmaker|dir\.?)\b", re.I)
HEAD = {"Authorization": f"Bearer {WIT_SERVER_TOKEN}"} if WIT_SERVER_TOKEN else {}

# =========================
# UI
# =========================
st.set_page_config(page_title="Netflix Movies Recommender Chatbot (Wit.ai)", page_icon="🎬", layout="wide")
st.title("🎬 Netflix Movies Recommender Chatbot (Wit.ai)")

if not WIT_SERVER_TOKEN:
    st.warning("No Wit.ai token found in Streamlit secrets (WIT_SERVER_TOKEN). Entity extraction will be limited; year-only still works.")

# =========================
# Wit helpers
# =========================
def wit_message(text: str) -> Dict[str, Any]:
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

def ent_any(data: Dict[str, Any], *names) -> Optional[str]:
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

def first_year(text: str) -> Optional[str]:
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None

def extract_years_any(data: Dict[str, Any], text: str) -> Tuple[Optional[str], Optional[str]]:
    v = ent_any(data, "year")
    if v and re.fullmatch(r"(?:19|20)\d{2}", str(v)):
        return str(v), str(v)
    y = first_year(text)
    return (y, y) if y else (None, None)

def is_pure_year(text: str) -> Optional[str]:
    s = text.strip()
    return s if re.fullmatch(r"(?:19|20)\d{2}", s) else None

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

def _rating_col(df: pd.DataFrame):
    for c in ["rating", "vote_average", "imdb_score"]:
        if c in df.columns:
            return c
    return None

def _load_df():
    df = pd.read_csv(CSV_PATH)
    for c in ["genres", "cast", "director"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_listish)
    return df

def call_list_movie(params: Dict[str, str]) -> pd.DataFrame:
    df = _load_df()

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

    if "genre" in params and "genres" in df.columns and params["genre"]:
        g = str(params["genre"]).lower()
        df = df[df["genres"].apply(lambda L: any(g == str(x).lower() for x in L))]

    if "actor" in params and "cast" in df.columns and params["actor"]:
        a = str(params["actor"]).lower()
        df = df[df["cast"].apply(lambda L: any(a in str(x).lower() for x in L))]

    if "director" in params and "director" in df.columns and params["director"]:
        d = str(params["director"]).lower()
        df = df[df["director"].apply(lambda L: any(d in str(x).lower() for x in L))]

    if "rating" in params and params["rating"]:
        try:
            n = int(params["rating"])
        except Exception:
            n = None
        if n and n > 0:
            rcol = _rating_col(df)
            df = (df.sort_values(rcol, ascending=False) if rcol else df).head(n)

    return df

def call_movie_info(title: str):
    df = _load_df()
    if "title" not in df.columns:
        return {}

    tnorm = str(title).lower().strip()
    df_titles = df["title"].astype(str).str.strip()

    # 1) exact match
    hit = df[df_titles.str.lower() == tnorm]

    # 2) substring match
    if hit.empty:
        hit = df[df_titles.str.lower().str.contains(tnorm)]

    # 3) fuzzy match
    if hit.empty:
        best = get_close_matches(tnorm, df_titles.str.lower().tolist(), n=1, cutoff=0.6)
        if best:
            hit = df[df_titles.str.lower() == best[0]]

    if hit.empty:
        return {}

    ycol = _pick_year_col(df)
    results = {}
    for _, row in hit.iterrows():
        results[str(row["title"])] = {
            "year_start": int(row[ycol]) if ycol and pd.notna(row[ycol]) else None,
            "directors": _to_listish(row["director"]) if "director" in df.columns else [],
            "actors": _to_listish(row["cast"]) if "cast" in df.columns else [],
            "genres": _to_listish(row["genres"]) if "genres" in df.columns else [],
            "rating": row[_rating_col(df)] if _rating_col(df) and pd.notna(row[_rating_col(df)]) else None,
        }
    return results

def call_movie_with_attribute(params: Dict[str, str]) -> pd.DataFrame:
    df = call_list_movie(params)
    attr = str(params.get("movie_attribute") or "").strip()
    if attr and attr in df.columns:
        s = df[attr]
        if s.dtype == "O":
            df = df[s.fillna("").astype(str).str.strip() != ""]
        else:
            df = df[s.notna()]
    return df

def _ents_for_base(data: Dict[str, Any], base_key: str):
    out = []
    ents = (data.get("entities") or {})
    for k, arr in ents.items():
        if k.split(":")[0] != base_key: 
            continue
        for e in (arr or []):
            v = e.get("value")
            if isinstance(v, dict): v = v.get("value")
            if v:
                out.append((int(e.get("start", -1)), int(e.get("end", -1)), str(v)))
    return out

def stitch_name_from_entities(data: Dict[str, Any], base_key: str, text: str) -> Optional[str]:
    pieces = _ents_for_base(data, base_key)
    if not pieces:
        return None
    pieces.sort(key=lambda t: (t[0] if t[0] is not None else 10**9))
    joined = " ".join(p[2] for p in pieces).strip()
    s = text.strip()
    if joined.replace(" ", "").lower() in s.replace(" ", "").lower():
        try:
            i = s.lower().find(pieces[0][2].lower())
            j = s.lower().rfind(pieces[-1][2].lower())
            if i != -1 and j != -1:
                return s[i:j+len(pieces[-1][2])].strip()
        except Exception:
            pass
    return joined

def guess_person_name(text: str) -> Optional[str]:
    s = text.strip()
    # 2–4 words, letters/space/basic punctuation
    if 1 < len(s.split()) <= 4 and re.fullmatch(r"[A-Za-z .'\-]+", s):
        return s
    return None

# =========================
# Display helpers
# =========================
def _join(lst, k):
    try:
        lst = lst or []
        return ", ".join([str(x) for x in lst[:k]])
    except Exception:
        return ""

def make_display_df(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    ycol = _pick_year_col(df)
    rcol = _rating_col(df)

    out = pd.DataFrame()
    out["Title"] = (df["title"].astype(str)
                    if "title" in df.columns else df.iloc[:, 0].astype(str))
    if ycol: out["Year"] = pd.to_numeric(df[ycol], errors="coerce").astype("Int64")
    if "genres" in df.columns:   out["Genres"]      = df["genres"].apply(lambda L: _join(L, 3))
    if "director" in df.columns: out["Director(s)"] = df["director"].apply(lambda L: _join(L, 2))
    if "cast" in df.columns:     out["Cast"]        = df["cast"].apply(lambda L: _join(L, 3))
    if rcol: out["Rating"] = pd.to_numeric(df[rcol], errors="coerce")

    cols = [c for c in ["Title", "Year", "Genres", "Director(s)", "Cast", "Rating"] if c in out.columns]
    out = out[cols].head(limit)
    return out

def summary_line(df: pd.DataFrame, params: Dict[str, str], limit: int) -> str:
    total = 0 if df is None else len(df)
    if total == 0:
        return "No movies found."

    shown = min(limit, total)

    pieces = []
    ys = params.get("year") or params.get("year_start") or params.get("release_year")
    ye = params.get("year_end") or ys
    if ys and ye and str(ys) == str(ye):
        pieces.append(f"from {ys}")
    elif ys and ye:
        pieces.append(f"from {ys}–{ye}")

    if params.get("genre"):
        pieces.append(f"in **{params['genre']}**")
    if params.get("director"):
        pieces.append(f"directed by **{params['director']}**")
    if params.get("actor"):
        pieces.append(f"starring **{params['actor']}**")
    if params.get("rating"):
        pieces.append(f"top **{params['rating']}**")

    spec = " ".join(pieces).strip()
    return f"Showing **{shown}** movies {spec}." if spec else f"Showing **{shown}** movies."

def _render_table(df: pd.DataFrame):
    if df is None or df.empty:
        return
    rcol = "Rating" if "Rating" in df.columns else None
    ycol = "Year"   if "Year"   in df.columns else None
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Title": st.column_config.TextColumn(width="large"),
            "Genres": st.column_config.TextColumn(width="medium"),
            "Director(s)": st.column_config.TextColumn(width="medium"),
            "Cast": st.column_config.TextColumn(width="large"),
            **({"Rating": st.column_config.NumberColumn(format="%.1f")} if rcol else {}),
            **({"Year": st.column_config.NumberColumn(format="%d")} if ycol else {}),
        },
    )


# =========================
# Formatters
# =========================
def format_list(response_json, prefix="Recommended movies are:", limit: Optional[int] = None):
    if not response_json:
        return "No movies found"

    # default limit
    L = RESULTS_LIMIT if (limit is None) else limit

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

    shown = response_json[:L]
    lines = [prefix]
    for i, item in enumerate(shown, 1):
        t = extract_title(item)
        if t:
            lines.append(f"{i}. {t}")

    remaining = max(0, len(response_json) - len(shown))
    if remaining > 0:
        lines.append(f"... and {remaining} more.")
    return "\n".join(lines) if len(lines) > 1 else "No movies found"

def pretty_title_info(response_json, want: str):
    if not response_json:
        return "Hmm… I couldn’t find that movie."

    out = []
    for title, payload in response_json.items():
        year = payload.get("year_start")
        directors = payload.get("directors") or []
        actors = payload.get("actors") or []
        genres = payload.get("genres") or []
        rating = payload.get("rating")

        if want == "year_start":
            out.append(f"🎬 **{title}** was released in {year}." if year else f"🎬 **{title}** – release year not available.")
        elif want == "directors":
            if directors:
                out.append(f"🎬 **{title}** was directed by {', '.join(directors)}.")
            else:
                out.append(f"🎬 **{title}** – I couldn’t find the director’s name.")
        elif want == "actors":
            if actors:
                out.append(f"🎬 **{title}** starred {', '.join(actors[:5])}.")
            else:
                out.append(f"🎬 **{title}** – I couldn’t find the cast list.")
        elif want == "genres":
            if genres:
                out.append(f"🎬 **{title}** falls under the genres: {', '.join(genres)}.")
            else:
                out.append(f"🎬 **{title}** – no genre info found.")
        elif want == "rating":
            if rating is not None:
                out.append(f"🎬 **{title}** has a rating of {rating}.")
            else:
                out.append(f"🎬 **{title}** – no rating available.")
        else:
            # Generic info fallback
            parts = []
            if year: parts.append(f"released in {year}")
            if directors: parts.append(f"directed by {', '.join(directors)}")
            if actors: parts.append(f"starring {', '.join(actors[:3])}")
            if genres: parts.append(f"in the {', '.join(genres)} genre")
            if rating is not None: parts.append(f"with a rating of {rating}")
            if parts:
                out.append(f"🎬 **{title}** was {', '.join(parts)}.")
            else:
                out.append(f"🎬 **{title}** – not much info available.")
    return "\n\n".join(out)

# =========================
# Router (returns: message, DataFrame|None)
# =========================
def route(text: str, data: Dict[str, Any]) -> Tuple[str, Optional[pd.DataFrame]]:
    intent = (data.get("intents") or [{}])[0].get("name")
    conf   = float((data.get("intents") or [{}])[0].get("confidence") or 0.0)

    director = stitch_name_from_entities(data, "director", text) or ent_any(data, "director")
    actor    = stitch_name_from_entities(data, "actor", text)    or ent_any(data, "actor")
    genre    = ent_any(data, "genre")
    title    = ent_any(data, "movie_title")
    attr_val = ent_any(data, "movie_attribute")
    y1, y2   = extract_years_any(data, text)
    ytxt     = is_pure_year(text)
    topn     = topn_from_text(text)
    limit    = max(RESULTS_LIMIT, _as_int(topn, RESULTS_LIMIT)) if topn else RESULTS_LIMIT
    name_guess = guess_person_name(text)
    if name_guess and not any([director, actor, genre, title, attr_val, y1, y2, ytxt]):
        # Prefer actor for plain name inputs like "Tika Sumpter"
        actor = name_guess

    if ytxt and not any([director, actor, genre, title, attr_val]):
        p = {"year": ytxt, "year_start": ytxt, "year_end": ytxt, "release_year": ytxt}
        df = call_list_movie(p)
        msg = summary_line(df, p, limit)
        return msg, make_display_df(df, limit)

    have_entities = any([director, actor, genre, title, attr_val, y1, y2, topn, ytxt])
    if (not intent or conf < 0.5) and not have_entities:
        return DEFAULT_FALLBACK, None

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
                df = call_list_movie(p)
                msg = summary_line(df, p, limit)
                return msg, make_display_df(df, limit)
            if director and not any([actor, genre, y1, title, attr_val]):
                if ACTORISH.search(text):
                    p_act = {"actor": director}
                    df_act = call_list_movie(p_act)
                    if not df_act.empty:
                        msg = summary_line(df_act, p_act, limit)
                        return msg, make_display_df(df_act, limit)
                p_dir = {"director": director}
                df_dir = call_list_movie(p_dir)
                if not df_dir.empty:
                    msg = summary_line(df_dir, p_dir, limit)
                    return msg, make_display_df(df_dir, limit)
                p_act = {"actor": director}
                df_act = call_list_movie(p_act)
                if not df_act.empty:
                    msg = summary_line(df_act, p_act, limit)
                    return msg, make_display_df(df_act, limit)
                return "No movies found.", None
            if actor and not any([director, genre, y1, title, attr_val]):
                if DIRECTORISH.search(text):
                    p_dir = {"director": actor}
                    df_dir = call_list_movie(p_dir)
                    if not df_dir.empty:
                        msg = summary_line(df_dir, p_dir, limit)
                        return msg, make_display_df(df_dir, limit)
                p_act = {"actor": actor}
                df_act = call_list_movie(p_act)
                if not df_act.empty:
                    msg = summary_line(df_act, p_act, limit)
                    return msg, make_display_df(df_act, limit)
                p_dir = {"director": actor}
                df_dir = call_list_movie(p_dir)
                if not df_dir.empty:
                    msg = summary_line(df_dir, p_dir, limit)
                    return msg, make_display_df(df_dir, limit)
                return "No movies found.", None
            if genre and not any([director, actor, y1, title, attr_val]):
                p = {"genre": genre}
                df = call_list_movie(p); msg = summary_line(df, p, limit)
                return msg, make_display_df(df, limit)
            if any([director, actor, genre, y1, ytxt]):
                y = y1 or ytxt
                p = {}
                if director: p["director"] = director
                if actor:    p["actor"]    = actor
                if genre:    p["genre"]    = genre
                if y:        p["year"] = p["year_start"] = y
                df = call_list_movie(p); msg = summary_line(df, p, limit)
                return msg, make_display_df(df, limit)

    if intent == "movie_match_director" and director:
        if ACTORISH.search(text) and not actor:
            p = {"actor": director}
            df = call_list_movie(p)
            if not df.empty:
                return summary_line(df, p, limit), make_display_df(df, limit)
        p_dir = {"director": director}
        df_dir = call_list_movie(p_dir)
        if not df_dir.empty:
            return summary_line(df_dir, p_dir, limit), make_display_df(df_dir, limit)
        if not actor:
            p_act = {"actor": director}
            df_act = call_list_movie(p_act)
            if not df_act.empty:
                return summary_line(df_act, p_act, limit), make_display_df(df_act, limit)
        return "No movies found.", None

    if intent == "movie_match_actor" and actor:
        if DIRECTORISH.search(text) and not director:
            p_dir = {"director": actor}
            df_dir = call_list_movie(p_dir)
            if not df_dir.empty:
                return summary_line(df_dir, p_dir, limit), make_display_df(df_dir, limit)
        p_act = {"actor": actor}
        df_act = call_list_movie(p_act)
        if not df_act.empty:
            return summary_line(df_act, p_act, limit), make_display_df(df_act, limit)
        if not director:
            p_dir = {"director": actor}
            df_dir = call_list_movie(p_dir)
            if not df_dir.empty:
                return summary_line(df_dir, p_dir, limit), make_display_df(df_dir, limit)
        return "No movies found.", None

    if intent == "movie_match_genre" and genre:
        p = {"genre": genre}
        df = call_list_movie(p); return summary_line(df, p, limit), make_display_df(df, limit)

    if intent == "movie_match_year" and (y1 or y2 or ytxt):
        y = y1 or ytxt
        p = {}
        if y:  p["year"] = p["year_start"] = y
        if y2: p["year_end"] = y2
        df = call_list_movie(p); return summary_line(df, p, limit), make_display_df(df, limit)

    if intent == "movie_match_rating" and title:
        info = call_movie_info(title)
        return pretty_title_info(info, want="rating"), None

    if intent == "movie_match_rating":
        p = {}
        if topn:     p["rating"]   = topn
        if genre:    p["genre"]    = genre
        if director: p["director"] = director
        if actor:    p["actor"]    = actor
        if (y1 or ytxt):
            y = y1 or ytxt
            p["year"] = p["year_start"] = y
        df = call_list_movie(p)
        lim = max(RESULTS_LIMIT, _as_int(topn, RESULTS_LIMIT)) if topn else RESULTS_LIMIT
        return summary_line(df, p, lim), make_display_df(df, lim)

    if intent == "movie_match_several_criteria":
        p = {}
        if director: p["director"] = director
        if actor:    p["actor"]    = actor
        if genre:    p["genre"]    = genre
        if (y1 or ytxt):
            y = y1 or ytxt
            p["year"] = p["year_start"] = y
        df = call_list_movie(p); return summary_line(df, p, limit), make_display_df(df, limit)

    if intent == "get_director_by_movie_title" and title:
        info = call_movie_info(title)
        return pretty_title_info(info, want="directors"), None
    if intent == "get_actor_by_movie_title" and title:
        info = call_movie_info(title)
        return pretty_title_info(info, want="actors"), None
    if intent == "get_year_by_movie_title" and title:
        info = call_movie_info(title)
        return pretty_title_info(info, want="year_start"), None
    if intent == "get_genre_by_movie_title" and title:
        info = call_movie_info(title)
        return pretty_title_info(info, want="genres"), None
    if intent == "get_rating_by_movie_title" and title:
        info = call_movie_info(title)
        return pretty_title_info(info, want="rating"), None
    if intent == "get_movie_attributes" and attr_val:
        p = {"movie_attribute": attr_val}
        if (y1 or ytxt):
            y = y1 or ytxt
            p["year"] = p["year_start"] = y
        df = call_movie_with_attribute(p); return summary_line(df, p, limit), make_display_df(df, limit)

    if (y1 or ytxt):
        y = y1 or ytxt
        p = {"year": y, "year_start": y, "year_end": y2 or y, "release_year": y}
        df = call_list_movie(p); return summary_line(df, p, limit), make_display_df(df, limit)

    return DEFAULT_FALLBACK, None

# =========================
# Chat Loop
# =========================
for m in st.session_state.get("history", []):
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("table"):
            df_prev = pd.DataFrame(m["table"], columns=m.get("columns"))
            _render_table(df_prev)

prompt = st.chat_input("Ask me about movies…")
if prompt:
    st.session_state.setdefault("history", []).append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        data = wit_message(prompt)
        message, table_df = route(prompt, data)
    except Exception as e:
        message, table_df = (f"Error: {e}", None)

    with st.chat_message("assistant"):
        st.markdown(message)
        hist_item = {"role": "assistant", "text": message}
        if isinstance(table_df, pd.DataFrame) and not table_df.empty:
            _render_table(table_df)
            hist_item["table"] = table_df.to_dict(orient="records")
            hist_item["columns"] = list(table_df.columns)

    st.session_state["history"].append(hist_item)
