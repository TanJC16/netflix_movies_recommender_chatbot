#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Wit.ai intents/entities on a JSON test set (no Streamlit imports).

Outputs (to console):
- Overall intent accuracy
- Entity micro-precision/recall/F1
- Per-intent accuracy table
- Per-entity recall
- Latency stats (avg/p50/p95)

Also writes a CSV of failing examples (text, expected vs predicted).

Environment:
  WIT_SERVER_TOKEN=...      (required)
  WIT_API_VERSION=20240901  (optional; default 20240901)

Usage:
  python test.py --json test_data.json --out wit_failures.csv
"""
import os, re, json, time, argparse, collections, csv, statistics
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
import requests

def get_secret(k: str, default: str = "") -> str:
    try:
        return st.secrets[k]
    except Exception:
        return os.getenv(k, default)
    
# ---------------- Config ----------------
WIT_SERVER_TOKEN = get_secret("WIT_SERVER_TOKEN", "")
API_VERSION      = os.getenv("WIT_API_VERSION", "20240901")
API_URL          = "https://api.wit.ai/message"

if not WIT_SERVER_TOKEN:
    raise SystemExit("ERROR: WIT_SERVER_TOKEN is not set. Export it and re-run.")

HEAD = {"Authorization": f"Bearer {WIT_SERVER_TOKEN}"}

# ---------------- Wit helpers ----------------
def wit_message(text: str) -> Dict[str, Any]:
    r = requests.get(
        API_URL,
        params={"q": text, "v": API_VERSION},
        headers=HEAD,
        timeout=15
    )
    r.raise_for_status()
    return r.json()

def _top_intent_name(payload: Dict[str, Any]) -> Optional[str]:
    intents = payload.get("intents") or []
    if not intents:
        return None
    return intents[0].get("name")

def _entities_as_pairs(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Map 'entity:role' -> str(value). If Wit lacks role, use 'entity:entity'.
    Only the first value per key is scored (aligns with your test JSON).
    """
    out = {}
    ents = payload.get("entities") or {}
    for k, arr in ents.items():
        if not isinstance(arr, list) or not arr:
            continue
        v = arr[0].get("value")
        if isinstance(v, dict):
            v = v.get("value")
        if v is None:
            continue
        key = k if ":" in k else f"{k}:{k}"
        out[key] = str(v)
    return out

# ---------------- Metrics helpers ----------------
def _norm_val(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z ]+", "", (s or "")).strip().lower()

def compare_entities(expected: Dict[str, str], predicted: Dict[str, str]) -> Tuple[int, int, int, Dict[str, Tuple[int,int]]]:
    """
    Return TP, FP, FN (micro) and per-entity-type (recall bookkeeping).
    We compare exact key ('entity:role') and normalized value.
    """
    per_type_hits: Dict[str, Tuple[int,int]] = collections.defaultdict(lambda: (0,0))

    exp = {(k, _norm_val(v)) for k, v in expected.items()}
    pred = {(k, _norm_val(v)) for k, v in predicted.items()}

    exp_keys = set(k for k,_ in exp)
    tp = sum(1 for pair in pred if pair in exp)
    fp = sum(1 for k,v in pred if (k in exp_keys) and ((k,v) not in exp))
    fn = sum(1 for pair in exp if pair not in pred)

    type_expected = collections.Counter([k.split(":")[0] for k,_ in exp])
    type_hits     = collections.Counter([k.split(":")[0] for k,v in pred if (k,v) in exp])

    for t in set(type_expected) | set(type_hits):
        per_type_hits[t] = (type_hits[t], type_expected[t])

    return tp, fp, fn, per_type_hits

# ---------------- Test loader ----------------
def load_cases(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = []
    for bucket, arr in data.items():
        if not isinstance(arr, list):
            continue
        for item in arr:
            cases.append({
                "bucket": bucket,
                "text": item.get("text", ""),
                "expected_intent": item.get("expected_intent"),
                "entities": {str(k): str(v) for k, v in (item.get("entities") or {}).items()}
            })
    return cases

# ---------------- Runner ----------------
# ... (imports and helpers stay the same)

def run_eval(json_path: str, out_csv: Optional[str], limit: Optional[int]):
    cases = load_cases(json_path)
    if limit:
        cases = cases[:limit]

    n = len(cases)
    if n == 0:
        raise SystemExit(f"ERROR: No cases loaded from {json_path}")

    intent_right = 0
    per_intent_tot = collections.Counter()
    per_intent_hit = collections.Counter()

    micro_tp = micro_fp = micro_fn = 0
    per_type_hits_total: Dict[str, List[int]] = collections.defaultdict(lambda: [0,0])
    latencies: List[float] = []
    failures = []

    for idx, c in enumerate(cases, 1):
        q = c["text"]
        t0 = time.time()
        try:
            resp = wit_message(q)
        except Exception as e:
            resp = {}
        dt = time.time() - t0
        latencies.append(dt)

        gold_intent = c["expected_intent"]
        pred_intent = _top_intent_name(resp)
        gold_ents = c["entities"]
        pred_ents = _entities_as_pairs(resp)

        tp, fp, fn, pertype = compare_entities(gold_ents, pred_ents)
        micro_tp += tp; micro_fp += fp; micro_fn += fn
        for t, (hit, tot) in pertype.items():
            agg = per_type_hits_total[t]
            agg[0] += hit; agg[1] += tot
            per_type_hits_total[t] = agg

        per_intent_tot[gold_intent] += 1
        if pred_intent == gold_intent:
            intent_right += 1
            per_intent_hit[gold_intent] += 1

        # progress log
        print(f"[{idx}/{n}] \"{q}\" | expected: {gold_intent}, predicted: {pred_intent}, "
              f"entities: {pred_ents}, time={dt:.2f}s")

        if (pred_intent != gold_intent) or (fn > 0) or (fp > 0):
            failures.append({
                "bucket": c["bucket"],
                "text": c["text"],
                "expected_intent": gold_intent,
                "pred_intent": pred_intent,
                "expected_entities": json.dumps(gold_ents, ensure_ascii=False),
                "pred_entities": json.dumps(pred_ents, ensure_ascii=False),
                "latency_s": round(dt, 3)
            })

    # Metrics
    intent_acc = intent_right / n
    precision  = (micro_tp / (micro_tp + micro_fp)) if (micro_tp + micro_fp) else 0.0
    recall     = (micro_tp / (micro_tp + micro_fn)) if (micro_tp + micro_fn) else 0.0
    f1         = (2*precision*recall / (precision + recall)) if (precision + recall) else 0.0

    avg_lat = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = (statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies))

    # Report
    print("===== Wit.ai Evaluation =====")
    print(f"Tests: {n}")
    print(f"Intent Accuracy: {intent_acc*100:.2f}%")
    print(f"Entity Precision: {precision*100:.2f}%  Recall: {recall*100:.2f}%  F1: {f1*100:.2f}%\n")

    print("Per-intent accuracy:")
    for intent, tot in sorted(per_intent_tot.items()):
        hit = per_intent_hit[intent]
        acc = hit / tot if tot else 0.0
        print(f"  {intent:30s}  {hit:3d}/{tot:<3d}  {acc*100:6.2f}%")

    if per_type_hits_total:
        print("\nPer-entity recall:")
        for etype, (hit, tot) in sorted(per_type_hits_total.items()):
            r = hit / tot if tot else 0.0
            print(f"  {etype:18s}  {hit:3d}/{tot:<3d}  {r*100:6.2f}%")

    print(f"\nLatency (s): avg {avg_lat:.3f} | p50 {p50:.3f} | p95 {p95:.3f}")

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["bucket","text","expected_intent","pred_intent","expected_entities","pred_entities","latency_s"]
            )
            writer.writeheader()
            writer.writerows(failures)
        print(f"\nSaved failing examples to: {out_csv}")

# ---------------- Main ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/mnt/data/test_data.json", help="Path to test JSON")
    ap.add_argument("--out", default="wit_failures.csv", help="CSV path to save failures")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of tests")
    args = ap.parse_args()

    run_eval(args.json, args.out, args.limit)
