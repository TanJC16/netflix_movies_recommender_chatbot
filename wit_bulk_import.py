
# wit_bulk_import.py
import os, json, time, requests, sys

# --- Wit.ai Server Access Token ---
WIT_TOKEN = "RDNAC35TZ5OV4VLUR43ACFW4CYPW65UP"
# ------------------------------------------------

if not WIT_TOKEN.strip():
    print("ERROR: please set WIT_TOKEN in this file.")
    sys.exit(1)

HEAD = {"Authorization": f"Bearer {WIT_TOKEN}", "Content-Type": "application/json"}

def get_json(url):
    r = requests.get(url, headers=HEAD, timeout=20)
    if not r.ok:
        print("HTTP", r.status_code, r.text[:500])
    r.raise_for_status()
    return r.json() if r.text else {}

def post_json(url, payload):
    r = requests.post(url, headers=HEAD, data=json.dumps(payload), timeout=20)
    if not r.ok:
        print("HTTP", r.status_code, r.text[:500])
    r.raise_for_status()
    return r.json() if r.text else {}

def put_json(url, payload):
    r = requests.put(url, headers=HEAD, data=json.dumps(payload), timeout=20)
    if not r.ok:
        print("HTTP", r.status_code, r.text[:500])
    r.raise_for_status()
    return r.json() if r.text else {}

def read_json_file(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _names_list(items):
    """Normalize API returns (strings or objects) to a set of names."""
    names = set()
    for x in items or []:
        if isinstance(x, str):
            names.add(x)
        elif isinstance(x, dict):
            n = x.get("name") or x.get("id") or x.get("entity")
            if n:
                names.add(n)
    return names

def create_intents(path="wit_intents.json"):
    data = read_json_file(path)
    if data is None:
        print(f"No {path} — skip")
        return
    existing = _names_list(get_json("https://api.wit.ai/intents"))
    for it in data:
        name = (it.get("name") or "").strip()
        if not name:
            continue
        if name in existing:
            print("Intent exists:", name)
            continue
        try:
            post_json("https://api.wit.ai/intents", {"name": name})
            print("Created intent:", name)
        except requests.HTTPError as e:
            msg = getattr(e.response, "text", "") or ""
            if "already exists" in msg.lower():
                print("Intent exists:", name)
            else:
                raise

def create_entities(path="wit_entities.json"):
    data = read_json_file(path)
    if data is None:
        print(f"No {path} — skip")
        return

    existing = _names_list(get_json("https://api.wit.ai/entities"))
    # Optional per-entity lookup override
    lookup_overrides = {
        # "genre": "keywords",   # uncomment to make 'genre' a keywords entity
    }

    for en in data:
        name = (en.get("name") or "").strip()
        if not name:
            continue
        if name.startswith("wit$"):
            print("Skip built-in:", name)
            continue

        lookups = [lookup_overrides.get(name, "free-text")]
        payload = {"name": name, "roles": [name], "lookups": lookups}

        if name in existing:
            # Update shape to be safe/consistent
            try:
                put_json(f"https://api.wit.ai/entities/{name}", payload)
                print("Updated entity:", name, "lookups=", lookups)
            except requests.HTTPError as e:
                msg = getattr(e.response, "text", "") or ""
                print("Entity update issue:", name, msg[:200])
            continue

        try:
            post_json("https://api.wit.ai/entities", payload)
            print("Created entity:", name, "lookups=", lookups)
        except requests.HTTPError as e:
            msg = getattr(e.response, "text", "") or ""
            if "already exists" in msg.lower():
                print("Entity exists:", name)
            else:
                raise

def upload_samples(path="wit_samples.json", batch=100):
    arr = read_json_file(path)
    if arr is None:
        print(f"No {path} — skip")
        return

    def canon_entity_id(name: str, role: str) -> str:
        """
        Build the 'entity' field for /utterances:
        - If name already contains a colon (e.g., 'genre:genre'), keep it as-is.
        - Otherwise, use 'name:role' (role defaults to name). Never stack colons.
        """
        name = (name or "").strip()
        role = (role or "").strip()
        if ":" in name:
            return name  # already canonical
        if role and ":" in role:
            role = role.split(":")[-1]  # keep only the role piece
        if not role:
            role = name
        return f"{name}:{role}"

    url = "https://api.wit.ai/utterances?v=20240901"

    for i in range(0, len(arr), batch):
        chunk = arr[i:i+batch]
        payload = []
        for s in chunk:
            ents = []
            for e in s.get("entities", []) or []:
                ent_id = canon_entity_id(e.get("entity"), e.get("role"))
                ents.append({
                    "entity": ent_id,
                    "start": int(e["start"]),
                    "end": int(e["end"]),
                    "body": e["body"],
                    "entities": e.get("entities", []) or [],
                })
            payload.append({
                "text": s["text"],
                "intent": s.get("intent"),
                "entities": ents,
                "traits": s.get("traits", []),
            })
        try:
            post_json(url, payload)
        except requests.HTTPError as e:
            # Help you pinpoint the offending item
            print("\nBatch failed for indices", i, "to", i + len(chunk) - 1)
            for idx, item in enumerate(payload):
                for ent in item.get("entities", []):
                    if ent["entity"].count(":") > 1:
                        print(" -> suspect at global index", i + idx,
                              "| text:", item["text"],
                              "| bad entity:", ent["entity"])
            raise
        print(f"Uploaded utterances {i+1}-{i+len(chunk)}")
        time.sleep(0.4)  # be polite

if __name__ == "__main__":
    create_intents("wit_intents.json")
    create_entities("wit_entities.json")
    upload_samples("wit_samples.json", batch=100)
    
    print("Done.")
