"""v2.8 — Map-reduce summarizer with agenda-aware talk segmentation,
exact tokenization, a SQLite run store, and knowledge-graph extraction.

Tuned for the 1050n (2 GB, Pascal sm_61, head) + 970w (4 GB, Maxwell sm_52,
RPC worker) llama.cpp cluster. Serving modes:
  serve-single: -c 16384, one slot (full 16K context per request)
  serve:        -c 16384 --parallel 2 (8,192 tokens per slot; pair with
                MAP_TOKENS=5000 / FINAL_TOKENS=6000 / MAP_WORKERS=2)

v2.8 fixes:
  * GraphML export: the networkx writer rejects None attribute values
    ("GraphML writer does not support <class 'NoneType'> as data values")
    — and sanitized KG fields are legitimately None sometimes (a node with
    no 'detail'). All graph attributes are now defaulted to "" before
    writing. (lxml being absent is only a benign writer fallback; lxml is
    now in the pixi deps for faster writes, but is not required.)
  * Export is never fatal: the final summary is printed BEFORE the KG file
    export, and the export itself is wrapped in try/except — a
    serialization failure can no longer eat the console output of a
    completed run.

v2.7: chat() fails fast on 4xx. v2.6: KG sanitizing; commit-before-KG.
v2.5: --kg-out auto; reextract-kg. v2.4: opt-in continuation; convergent
reduce. v2.3: finish_reason handling; kg tables; per-summary metadata.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sqlite3
import statistics
import sys
import time

import requests

BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8196")
MODEL = os.environ.get("LLAMA_MODEL", "qwen3-4b-instruct-2507")
API = BASE_URL.rstrip("/") + "/v1/chat/completions"
TOKENIZE = BASE_URL.rstrip("/") + "/tokenize"
DB_PATH = os.environ.get("SUMMDB", "summaries.db")
SCRIPT_VERSION = "2.8"

MAP_TOKENS = int(os.environ.get("MAP_TOKENS", "6000"))
OVERLAP_TOKENS = int(os.environ.get("OVERLAP_TOKENS", "150"))
FINAL_TOKENS = int(os.environ.get("FINAL_TOKENS", "8000"))
NIAH_CTX_TOKENS = int(os.environ.get("NIAH_CTX_TOKENS", "12000"))
MAP_WORKERS = int(os.environ.get("MAP_WORKERS", "1"))
SEG_TOL_S = int(os.environ.get("SEG_TOL_S", "600"))
TITLE_W = 0.55


class Trunc:
    hit = False


# --------------------------------------------------------------------------
# LLM plumbing (continuation is opt-in: max_continues=0 by default)
# --------------------------------------------------------------------------
def chat(system: str, user: str, max_tokens: int = 700, temp: float = 0.2,
         retries: int = 5, max_continues: int = 0) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    parts = []
    for _ in range(max_continues + 1):
        payload = {"model": MODEL, "messages": messages,
                   "temperature": temp, "max_tokens": max_tokens}
        content = None
        for attempt in range(retries):
            try:
                r = requests.post(API, json=payload, timeout=1800)
                r.raise_for_status()
                choice = r.json()["choices"][0]
                content = choice["message"]["content"].strip()
                finish = choice.get("finish_reason", "stop")
                break
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if 400 <= status < 500:
                    msg = str(e)
                    try:
                        msg = e.response.json().get("error", {}).get("message", msg)
                    except ValueError:
                        pass
                    raise RuntimeError(
                        f"request rejected by server ({status}): {msg}")
                wait = min(2 ** attempt * 2, 60)
                print(f"  chat error ({e}); retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
            except (requests.RequestException, KeyError, IndexError) as e:
                wait = min(2 ** attempt * 2, 60)
                print(f"  chat error ({e}); retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
        if content is None:
            raise RuntimeError("chat() failed after retries")
        parts.append(content)
        if finish != "length" or max_continues == 0:
            if finish == "length":
                print("  [hit max_tokens; output truncated at cap]",
                      file=sys.stderr)
                Trunc.hit = True
            return "\n".join(parts)
        Trunc.hit = True
        print(f"  [finish_reason=length, continuing "
              f"({len(parts)} part(s) so far)]", file=sys.stderr)
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content":
                         "Continue EXACTLY where your previous message "
                         "stopped. Do not repeat earlier text, do not "
                         "restart, keep the same format."})
    return "\n".join(parts)


def count_tokens(text: str) -> int:
    try:
        r = requests.post(TOKENIZE, json={"content": text}, timeout=120)
        r.raise_for_status()
        return len(r.json()["tokens"])
    except requests.RequestException:
        return len(text) // 4


def ntok_to_chars(n: int) -> int:
    return n * 4


def _trim_to_tokens(text: str, max_tok: int) -> str:
    if count_tokens(text) <= max_tok:
        return text
    while count_tokens(text) > max_tok and len(text) > 200:
        cut = int(len(text) * 0.9)
        pos = text.rfind(". ", 0, cut)
        text = text[: pos + 1] if pos > len(text) // 2 else text[:cut]
    return text


# --------------------------------------------------------------------------
# Text extraction & token-exact chunking
# --------------------------------------------------------------------------
def extract_text(pdf_path: str) -> str:
    import fitz

    doc = fitz.open(pdf_path)
    text = "\n\n".join(page.get_text() for page in doc)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_tokens: int, overlap_tokens: int):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + ntok_to_chars(max_tokens), len(text))
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + ntok_to_chars(max_tokens) // 2:
                end = boundary + 1
        piece = text[start:end]
        while count_tokens(piece) > max_tokens and len(piece) > 200:
            piece = piece[: int(len(piece) * 0.95)]
        chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - ntok_to_chars(overlap_tokens), start + 1)
    return chunks


# --------------------------------------------------------------------------
# Agenda parsing & transcript segmentation
# --------------------------------------------------------------------------
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")


def parse_agenda(path: str):
    agenda = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = TIME_RE.search(line)
            if not m:
                continue
            h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
            start = h * 3600 + mi * 60 + s
            title = line[m.end():].strip(" ,;-—\t")
            if title:
                agenda.append({"start": start, "title": title})
    agenda.sort(key=lambda a: a["start"])
    return agenda


def load_whisper_json(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    segs = data["segments"] if isinstance(data, dict) else data
    return [{"start": float(s["start"]), "end": float(s["end"]),
             "text": s["text"].strip()} for s in segs]


def _title_similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    try:
        from rapidfuzz import fuzz
        return fuzz.partial_ratio(a, b) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a[:200], b[:2000]).ratio()


def _boundary_candidates(segs, target: float, tol: float):
    out = []
    for i, s in enumerate(segs):
        if abs(s["start"] - target) <= tol:
            gap = (s["start"] - segs[i - 1]["end"]) if i else 5.0
            out.append((s["start"], i, gap))
    return out


def segment_talks(segs, agenda, tol=SEG_TOL_S, passes=3):
    if not segs or not agenda:
        return []
    offset = 0.0
    chosen = []
    for _ in range(passes):
        picks = []
        for item in agenda:
            sched = item["start"] + offset
            cands = _boundary_candidates(segs, sched, tol)
            if not cands:
                nearest = min(segs, key=lambda s: abs(s["start"] - sched))
                cands = [(nearest["start"], segs.index(nearest), 0.0)]
            best, best_score = None, -1.0
            for t, idx, gap in cands:
                time_score = max(0.0, 1.0 - abs(t - sched) / tol)
                text_after = " ".join(
                    s["text"] for s in segs[idx: idx + 25])[:2000]
                title_score = _title_similarity(item["title"], text_after)
                pause_score = min(gap / 5.0, 1.0)
                score = ((1 - TITLE_W) * time_score + TITLE_W * title_score
                         + 0.1 * pause_score)
                if score > best_score:
                    best, best_score = (t, idx), score
            picks.append((best[0], best[1], item))
        fixed, last_idx = [], -1
        for t, idx, item in picks:
            if idx <= last_idx:
                idx = last_idx + 1
                if idx >= len(segs):
                    continue
                t = segs[idx]["start"]
            fixed.append((t, idx, item))
            last_idx = idx
        chosen = fixed
        offsets = [t - item["start"] for t, _, item in chosen if t is not None]
        if offsets:
            new_offset = statistics.median(offsets)
            if abs(new_offset - offset) < 1.0:
                break
            offset = new_offset
    talks = []
    for k, (t, idx, item) in enumerate(chosen):
        end_idx = chosen[k + 1][1] if k + 1 < len(chosen) else len(segs)
        body = segs[idx:end_idx]
        if body:
            talks.append({
                "title": item["title"], "sched_start": item["start"],
                "seg_start": t, "text": " ".join(s["text"] for s in body),
                "n_segs": len(body), "clock_offset": t - item["start"],
            })
    return talks


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
MAP_SYSTEM = (
    "You are a meticulous technical summarizer. You extract EVERY substantive "
    "fact, number, definition, argument, and conclusion from document fragments. "
    "You never invent content. Output 5-9 terse bullet points. Each bullet is "
    "one line. Be dense, not verbose."
)
REDUCE_SYSTEM = (
    "You are a technical editor condensing document digests. Merge them into a "
    "denser outline. Preserve ALL facts, numbers, and conclusions; only remove "
    "redundancy. Output tight bullet points, at most one screen. Your merged "
    "digest MUST be shorter than the combined input digests."
)
FINAL_SYSTEM = (
    "You are an expert technical writer producing the final executive summary "
    "of a document. Be comprehensive but readable: an overview paragraph, then "
    "key findings, methods/data, and conclusions as bullets. Cite section "
    "positions (early/middle/late) when relevant. Finish with a complete "
    "closing sentence."
)
TALK_FINAL_SYSTEM = (
    "You are an expert biomedical research writer. Produce a structured summary "
    "of ONE conference talk: (1) one-paragraph abstract, (2) key findings, "
    "(3) methods/data, (4) conclusions/open questions. Preserve all numbers, "
    "model names, and caveats. Never invent content."
)
KG_SYSTEM = (
    "You are an information-extraction engine. You output ONLY valid JSON, "
    "no prose and no code fences. Extract a knowledge graph from the given "
    "text. At most 40 nodes and at most 60 edges. Nodes are entities (types "
    "such as: protein, gene, disease, drug, method, dataset, metric, "
    "statistic, claim, organization, model). Edges are typed relations "
    "between existing nodes (such as: activates, inhibits, measured_by, "
    "compared_to, supports, contradicts, associated_with, used_in, targets). "
    "Node keys are short snake_case identifiers. Every 'detail' and every "
    "'evidence' MUST be a plain string of a few words, never a nested object "
    "or list. Keep every 'detail' under 15 words and every 'evidence' under "
    "12 words. Output the JSON and nothing else, and make sure it is complete "
    "and valid before finishing."
)


def map_step(items, label="map"):
    summaries = [None] * len(items)

    def one(i):
        chunk = items[i]
        user = (
            f"Summarize the following fragment ({i + 1} of {len(items)}) of a "
            f"larger document. Capture every distinct fact, entity, number, and "
            f"claim. Note explicitly if the fragment is an introduction, body, or "
            f"conclusion section.\n\n--- DOCUMENT FRAGMENT {i + 1}/{len(items)} "
            f"---\n{chunk}"
        )
        out = chat(MAP_SYSTEM, user, max_tokens=1100, max_continues=1)
        print(f"  {label} {i + 1}/{len(items)} done "
              f"({count_tokens(chunk)} tok in)", file=sys.stderr)
        return i, out

    if MAP_WORKERS > 1:
        with cf.ThreadPoolExecutor(MAP_WORKERS) as ex:
            for i, out in ex.map(one, range(len(items))):
                summaries[i] = out
    else:
        for i in range(len(items)):
            i2, out = one(i)
            summaries[i2] = out
    return summaries


def reduce_step(summaries, max_level=8):
    """Token-exact, provably convergent hierarchical reduce."""
    target = max(FINAL_TOKENS // 2, 1000)
    batch_tok = max(target // 2, 400)
    layer = [s for s in summaries if s and s.strip()]
    level = 0

    def total_tok():
        return sum(count_tokens(s) for s in layer)

    while total_tok() > target:
        level += 1
        if level > max_level:
            layer = [_trim_to_tokens(s, max(batch_tok // 2, 200))
                     for s in layer]
            print(f"  reduce: hit max_level {max_level}; force-trimmed to "
                  f"{total_tok()} tokens", file=sys.stderr)
            break
        batches, current = [], []
        for s in layer:
            current.append(s)
            if len(current) == 3:
                batches.append(current)
                current = []
        if current:
            if len(current) == 1 and batches:
                batches[-1].extend(current)
            else:
                batches.append(current)
        nxt = []
        for b in batches:
            if len(b) == 1:
                merged = b[0]
            else:
                merged = chat(REDUCE_SYSTEM,
                              "Digests to merge:\n\n" + "\n\n".join(b),
                              max_tokens=1200)
            nxt.append(_trim_to_tokens(merged, batch_tok))
        layer = nxt
        print(f"  reduce level {level}: {len(layer)} digests, "
              f"{total_tok()} tokens total", file=sys.stderr)
    return layer


def final_step(digests, name, system=FINAL_SYSTEM):
    outline = "\n\n".join(f"[Section {i + 1} digest]\n{d}" for i, d in enumerate(digests))
    user = (
        f"The digests below cover the ENTIRE document '{name}' in reading order "
        f"(section 1 = document beginning). Synthesize them into the final "
        f"summary. Do not omit facts that appear in only one section.\n\n"
        f"{outline}\n\nProduce the final summary now."
    )
    return chat(system, user, max_tokens=1600, max_continues=2)


# --------------------------------------------------------------------------
# Knowledge-graph extraction (sanitized, with JSON repair + salvage)
# --------------------------------------------------------------------------
def _clean_str(v):
    """Coerce an LLM-emitted field to a safe str (or None)."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v or None
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(v)


def _repair_json(raw: str) -> str:
    stack, in_str, esc = [], False, False
    for ch in raw:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if stack:
                    stack.pop()
    fix = raw + ('"' if in_str else "")
    closer = {"[": "]", "{": "}"}
    for ch in reversed(stack):
        fix += closer[ch]
    return fix


def _salvage_objects(raw: str):
    def objs(seg):
        out = []
        for mo in re.finditer(r"\{[^{}]*\}", seg or ""):
            try:
                out.append(json.loads(mo.group(0)))
            except json.JSONDecodeError:
                pass
        return out

    nodes, edges = [], []
    m_nodes = re.search(r'"nodes"\s*:\s*\[', raw)
    m_edges = re.search(r'"edges"\s*:\s*\[', raw)
    if m_nodes:
        end = (m_edges.start() if m_edges and m_edges.start() > m_nodes.start()
               else len(raw))
        nodes = objs(raw[m_nodes.end():end])
    if m_edges:
        edges = objs(raw[m_edges.end():])
    return nodes, edges


def extract_kg(text, retries=2):
    user = (
        'Extract the knowledge graph as JSON with exactly this shape: '
        '{"nodes":[{"key":"snake_case_id","label":"Human readable",'
        '"type":"entity type","detail":"few words"}],'
        '"edges":[{"source":"node key","target":"node key",'
        '"relation":"relation type","evidence":"short quote"}]}\n\n'
        f"--- TEXT ---\n{text}"
    )
    last_raw = ""
    for _ in range(retries):
        raw = chat(KG_SYSTEM, user, max_tokens=3000, temp=0.0, max_continues=2)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.M)
        last_raw = raw
        for candidate in (raw, _repair_json(raw)):
            try:
                data = json.loads(candidate)
                nodes, edges = data.get("nodes", []), data.get("edges", [])
                if isinstance(nodes, list) and isinstance(edges, list):
                    return _dedupe_kg(nodes, edges)
            except json.JSONDecodeError:
                continue
    nodes, edges = _salvage_objects(last_raw)
    if nodes or edges:
        print(f"  kg: salvaged partial graph ({len(nodes)} nodes, "
              f"{len(edges)} edges) from truncated JSON", file=sys.stderr)
        return _dedupe_kg(nodes, edges)
    print("  kg extraction: JSON parse failed", file=sys.stderr)
    return [], []


def _dedupe_kg(nodes, edges):
    """Sanitize every field, drop malformed entries, dedupe nodes/edges."""
    seen, out_n = set(), []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        key = _clean_str(n.get("key")) or _clean_str(n.get("label"))
        if not key:
            continue
        k = key.lower()
        if k in seen:
            continue
        seen.add(k)
        out_n.append({
            "key": key,
            "label": _clean_str(n.get("label")) or key,
            "type": _clean_str(n.get("type")) or "entity",
            "detail": _clean_str(n.get("detail")),
        })
    keys = {n["key"] for n in out_n}
    out_e, seen_e = [], set()
    for e in edges:
        if not isinstance(e, dict):
            continue
        src, tgt = _clean_str(e.get("source")), _clean_str(e.get("target"))
        rel = _clean_str(e.get("relation"))
        if not src or not tgt or not rel or src not in keys or tgt not in keys:
            continue
        sig = (src, tgt, rel)
        if sig in seen_e:
            continue
        seen_e.add(sig)
        out_e.append({"source": src, "target": tgt, "relation": rel,
                      "evidence": _clean_str(e.get("evidence"))})
    return out_n, out_e


def store_kg(conn, doc_id, talk_id, nodes, edges):
    for n in nodes:
        if not isinstance(n, dict):
            continue
        conn.execute(
            "INSERT INTO kg_nodes(doc_id, talk_id, node_key, label, type, detail) "
            "VALUES(?,?,?,?,?,?)",
            (doc_id, talk_id, _clean_str(n.get("key")),
             _clean_str(n.get("label")), _clean_str(n.get("type")),
             _clean_str(n.get("detail"))))
    for e in edges:
        if not isinstance(e, dict):
            continue
        conn.execute(
            "INSERT INTO kg_edges(doc_id, talk_id, source_key, target_key, "
            "relation, evidence) VALUES(?,?,?,?,?,?)",
            (doc_id, talk_id, _clean_str(e.get("source")),
             _clean_str(e.get("target")), _clean_str(e.get("relation")),
             _clean_str(e.get("evidence"))))
    return len(nodes), len(edges)


def kg_row_tuples(nodes, edges, talk_id=None):
    n_rows = [(talk_id, _clean_str(n.get("key")), _clean_str(n.get("label")),
               _clean_str(n.get("type")), _clean_str(n.get("detail")))
              for n in nodes if isinstance(n, dict)]
    e_rows = [(talk_id, _clean_str(e.get("source")), _clean_str(e.get("target")),
               _clean_str(e.get("relation")), _clean_str(e.get("evidence")))
              for e in edges if isinstance(e, dict)]
    return n_rows, e_rows


def _resolve_kg_out(value, input_path):
    if value != "auto":
        return value
    stem = os.path.splitext(os.path.basename(input_path))[0]
    try:
        import networkx  # noqa: F401
        return f"{stem}.kg.graphml"
    except ImportError:
        return f"{stem}.kg.json"


def _gval(v, default=""):
    """GraphML writers reject None attribute values — default them."""
    return default if v is None else v


def _write_kg_file(nodes, edges, meta, out):
    """Serialize KG rows (tuple shape) to .json / .graphml / .dot.
    meta = (model, script_version, path, sha256). No LLM calls."""
    if out.endswith(".graphml"):
        try:
            import networkx as nx

            g = nx.DiGraph()
            for n in nodes:
                if n[1]:
                    g.add_node(n[1], label=_gval(n[2]), type=_gval(n[3], "entity"),
                               detail=_gval(n[4]), talk_id=_gval(n[0]))
            for e in edges:
                if g.has_node(e[1]) and g.has_node(e[2]):
                    g.add_edge(e[1], e[2], relation=_gval(e[3]),
                               evidence=_gval(e[4]), talk_id=_gval(e[0]))
            nx.write_graphml(g, out)
            print(f"  kg file: {out} ({g.number_of_nodes()} nodes, "
                  f"{g.number_of_edges()} edges, model: {meta[0]})",
                  file=sys.stderr)
            return
        except ImportError:
            print("networkx not installed; writing JSON instead",
                  file=sys.stderr)
            out = out.replace(".graphml", ".json")
    elif out.endswith(".dot"):
        def q(s):
            return '"%s"' % str(s).replace('"', "'")

        with open(out, "w", encoding="utf-8") as f:
            f.write("digraph kg {\n")
            for n in nodes:
                f.write(f"  {q(n[1])} [label={q(n[2] + ' [' + str(n[3]) + ']')}];\n")
            for e in edges:
                f.write(f"  {q(e[1])} -> {q(e[2])} [label={q(e[3])}];\n")
            f.write("}\n")
        print(f"  kg file: {out} (model: {meta[0]})", file=sys.stderr)
        return
    payload = {
        "model": meta[0], "script_version": meta[1], "path": meta[2],
        "sha256": meta[3],
        "nodes": [{"talk_id": n[0], "key": n[1], "label": n[2],
                   "type": n[3], "detail": n[4]} for n in nodes],
        "edges": [{"talk_id": e[0], "source": e[1], "target": e[2],
                   "relation": e[3], "evidence": e[4]} for e in edges],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  kg file: {out} ({len(nodes)} nodes, {len(edges)} edges, "
          f"model: {meta[0]})", file=sys.stderr)


# --------------------------------------------------------------------------
# SQLite run store
# --------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY, ts TEXT DEFAULT (datetime('now')),
  model TEXT, params_json TEXT, script_version TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES runs(id),
  kind TEXT, path TEXT, sha256 TEXT, n_tokens INTEGER);
CREATE TABLE IF NOT EXISTS talks(
  id INTEGER PRIMARY KEY, doc_id INTEGER REFERENCES documents(id),
  idx INTEGER, title TEXT, sched_start REAL, seg_start REAL,
  clock_offset REAL, n_tokens INTEGER);
CREATE TABLE IF NOT EXISTS summaries(
  id INTEGER PRIMARY KEY, doc_id INTEGER REFERENCES documents(id),
  talk_id INTEGER REFERENCES talks(id), stage TEXT, text TEXT, elapsed REAL,
  model TEXT, truncated INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS kg_nodes(
  id INTEGER PRIMARY KEY, doc_id INTEGER REFERENCES documents(id),
  talk_id INTEGER REFERENCES talks(id), node_key TEXT, label TEXT,
  type TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS kg_edges(
  id INTEGER PRIMARY KEY, doc_id INTEGER REFERENCES documents(id),
  talk_id INTEGER REFERENCES talks(id), source_key TEXT, target_key TEXT,
  relation TEXT, evidence TEXT);
CREATE INDEX IF NOT EXISTS idx_summ_doc ON summaries(doc_id, stage);
CREATE INDEX IF NOT EXISTS idx_kg_doc ON kg_nodes(doc_id);
CREATE INDEX IF NOT EXISTS idx_kge_doc ON kg_edges(doc_id);
"""

MIGRATIONS = [
    "ALTER TABLE summaries ADD COLUMN model TEXT",
    "ALTER TABLE summaries ADD COLUMN truncated INTEGER DEFAULT 0",
]


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    for stmt in MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def start_run(conn, source: str, params: dict) -> int:
    cur = conn.execute(
        "INSERT INTO runs(model, params_json, script_version, source) VALUES(?,?,?,?)",
        (MODEL, json.dumps(params, sort_keys=True), SCRIPT_VERSION, source))
    return cur.lastrowid


def insert_summary(conn, doc_id, talk_id, stage, text, elapsed):
    conn.execute(
        "INSERT INTO summaries(doc_id, talk_id, stage, text, elapsed, model, "
        "truncated) VALUES(?,?,?,?,?,?,?)",
        (doc_id, talk_id, stage, text, elapsed, MODEL,
         1 if Trunc.hit else 0))
    Trunc.hit = False


# --------------------------------------------------------------------------
# CLI commands
# --------------------------------------------------------------------------
def cmd_summarize_pdf(args, conn):
    run_id = start_run(conn, "pdf", {
        "map_tokens": MAP_TOKENS, "overlap": OVERLAP_TOKENS,
        "final_tokens": FINAL_TOKENS, "pdf": args.pdf})
    print(f"Extracting {args.pdf} ... (run {run_id})", file=sys.stderr)
    text = extract_text(args.pdf)
    chunks = chunk_text(text, MAP_TOKENS, OVERLAP_TOKENS)
    n_tokens = count_tokens(text)
    print(f"{n_tokens} tokens -> {len(chunks)} chunks", file=sys.stderr)
    sha = sha256_file(args.pdf)
    cur = conn.execute(
        "INSERT INTO documents(run_id, kind, path, sha256, n_tokens) VALUES(?,?,?,?,?)",
        (run_id, "pdf", args.pdf, sha, n_tokens))
    doc_id = cur.lastrowid
    summaries = map_step(chunks)
    digests = reduce_step(summaries)
    final = final_step(digests, os.path.basename(args.pdf))
    insert_summary(conn, doc_id, None, "final", final, 0)
    for stage, items in (("chunk_digest", summaries), ("reduced_digest", digests)):
        for s in items:
            insert_summary(conn, doc_id, None, stage, s, 0)
    # Commit BEFORE KG extraction: a KG failure must never roll back the run.
    conn.commit()
    nodes, edges = [], []
    try:
        nodes, edges = extract_kg("\n\n".join(digests))
        nn, ne = store_kg(conn, doc_id, None, nodes, edges)
        conn.commit()
        print(f"  knowledge graph: {nn} nodes, {ne} edges", file=sys.stderr)
    except Exception as e:
        conn.rollback()
        print(f"  knowledge graph: FAILED ({e}); summaries already "
              f"committed, use 'reextract-kg' later", file=sys.stderr)
    # Print the final summary BEFORE any optional file export, and make the
    # export non-fatal: it must never eat a completed run's console output.
    print("\n" + "=" * 70 + f"\nFINAL SUMMARY (run {run_id})\n" + "=" * 70)
    print(final)
    if args.kg_out:
        try:
            n_rows, e_rows = kg_row_tuples(nodes, edges)
            _write_kg_file(n_rows, e_rows,
                           (MODEL, SCRIPT_VERSION, args.pdf, sha),
                           _resolve_kg_out(args.kg_out, args.pdf))
        except Exception as e:
            print(f"  kg file: FAILED ({e}); the graph remains stored in "
                  f"{DB_PATH} — retry with 'export-kg'", file=sys.stderr)


def cmd_segment(args, conn):
    run_id = start_run(conn, "talks", {
        "agenda": args.agenda, "tolerance_s": SEG_TOL_S,
        "map_tokens": MAP_TOKENS})
    segs = load_whisper_json(args.whisper_json)
    agenda = parse_agenda(args.agenda)
    print(f"{len(agenda)} agenda items, {len(segs)} transcript segments",
          file=sys.stderr)
    talks = segment_talks(segs, agenda)
    sha = sha256_file(args.whisper_json)
    cur = conn.execute(
        "INSERT INTO documents(run_id, kind, path, sha256, n_tokens) VALUES(?,?,?,?,?)",
        (run_id, "talks", args.whisper_json, sha,
         sum(count_tokens(t["text"]) for t in talks)))
    doc_id = cur.lastrowid
    all_n_rows, all_e_rows = [], []
    for i, t in enumerate(talks):
        print(f"[{i + 1}/{len(talks)}] {t['title']} "
              f"(sched {t['sched_start']:.0f}s -> found {t['seg_start']:.0f}s, "
              f"offset {t['clock_offset']:+.0f}s)", file=sys.stderr)
        cur = conn.execute(
            "INSERT INTO talks(doc_id, idx, title, sched_start, seg_start, "
            "clock_offset, n_tokens) VALUES(?,?,?,?,?,?,?)",
            (doc_id, i, t["title"], t["sched_start"], t["seg_start"],
             t["clock_offset"], count_tokens(t["text"])))
        talk_id = cur.lastrowid
        t0 = time.time()
        chunks = chunk_text(t["text"], MAP_TOKENS, OVERLAP_TOKENS)
        digests = map_step(chunks, label=f"map[{t['title'][:30]}]")
        if len(digests) > 1:
            digests = reduce_step(digests)
        final = final_step(digests, t["title"], system=TALK_FINAL_SYSTEM)
        insert_summary(conn, doc_id, talk_id, "talk_final", final, time.time() - t0)
        conn.commit()
        try:
            nodes, edges = extract_kg("\n\n".join(digests))
            nn, ne = store_kg(conn, doc_id, talk_id, nodes, edges)
            conn.commit()
            print(f"  kg: {nn} nodes, {ne} edges", file=sys.stderr)
        except Exception as e:
            conn.rollback()
            print(f"  kg: FAILED for talk {i + 1} ({e}); talk summary "
                  f"already committed", file=sys.stderr)
            continue
        if args.kg_out:
            n_rows, e_rows = kg_row_tuples(nodes, edges, talk_id=talk_id)
            all_n_rows.extend(n_rows)
            all_e_rows.extend(e_rows)
        print(final + "\n" + "-" * 70)
    if args.kg_out and all_n_rows:
        try:
            _write_kg_file(all_n_rows, all_e_rows,
                           (MODEL, SCRIPT_VERSION, args.whisper_json, sha),
                           _resolve_kg_out(args.kg_out, args.whisper_json))
        except Exception as e:
            print(f"  kg file: FAILED ({e}); graphs remain stored in "
                  f"{DB_PATH} — retry with 'export-kg'", file=sys.stderr)
    print(f"run {run_id}: {len(talks)} talks stored in {DB_PATH}",
          file=sys.stderr)


def cmd_runs(args, conn):
    for row in conn.execute(
            "SELECT r.id, r.ts, r.model, r.source, r.params_json, "
            "COUNT(DISTINCT d.id), COUNT(s.id) FROM runs r "
            "LEFT JOIN documents d ON d.run_id = r.id "
            "LEFT JOIN summaries s ON s.doc_id = d.id "
            "GROUP BY r.id ORDER BY r.id DESC"):
        print(f"run {row[0]} | {row[1]} | {row[2]} | {row[3]} | "
              f"docs={row[5]} summaries={row[6]}")
        print(f"  params: {row[4]}")


def cmd_compare(args, conn):
    for row in conn.execute(
            "SELECT r.id, s.model, d.path, s.stage, s.truncated, "
            "substr(s.text,1,300) FROM summaries s "
            "JOIN documents d ON s.doc_id=d.id JOIN runs r ON d.run_id=r.id "
            "WHERE s.stage=? ORDER BY r.id", (args.stage,)):
        print(f"\n=== run {row[0]} | {row[1]} | {row[2]} | "
              f"truncated={row[4]} ===\n{row[5]}...")


def _resolve_doc(conn, ref: str) -> int:
    if ref.isdigit():
        row = conn.execute("SELECT id FROM documents WHERE id=?",
                           (int(ref),)).fetchone()
    else:
        row = conn.execute(
            "SELECT id FROM documents WHERE path LIKE ? OR sha256=? "
            "ORDER BY id DESC LIMIT 1", (f"%{ref}%", ref)).fetchone()
    if not row:
        raise SystemExit(f"no document matches '{ref}'")
    return row[0]


def _fetch_kg_rows(conn, doc_id):
    nodes = conn.execute(
        "SELECT talk_id, node_key, label, type, detail FROM kg_nodes "
        "WHERE doc_id=?", (doc_id,)).fetchall()
    edges = conn.execute(
        "SELECT talk_id, source_key, target_key, relation, evidence "
        "FROM kg_edges WHERE doc_id=?", (doc_id,)).fetchall()
    return nodes, edges


def cmd_export_kg(args, conn):
    doc_id = _resolve_doc(conn, args.doc)
    meta = conn.execute(
        "SELECT r.model, r.script_version, d.path, d.sha256 FROM documents d "
        "JOIN runs r ON d.run_id=r.id WHERE d.id=?", (doc_id,)).fetchone()
    nodes, edges = _fetch_kg_rows(conn, doc_id)
    _write_kg_file(nodes, edges, meta, args.out)


def cmd_reextract_kg(args, conn):
    """Rebuild the KG for an already-summarized document from its STORED
    digests — one extraction pass, no PDF parse, no map/reduce."""
    doc_id = _resolve_doc(conn, args.doc)
    meta = conn.execute(
        "SELECT r.model, r.script_version, d.path, d.sha256 FROM documents d "
        "JOIN runs r ON d.run_id=r.id WHERE d.id=?", (doc_id,)).fetchone()
    rows = conn.execute(
        "SELECT text FROM summaries WHERE doc_id=? AND stage='reduced_digest' "
        "ORDER BY id", (doc_id,)).fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT text FROM summaries WHERE doc_id=? AND stage='chunk_digest' "
            "ORDER BY id", (doc_id,)).fetchall()
    if not rows:
        raise SystemExit(f"no stored digests for document '{args.doc}'; "
                         f"run summarize first")
    text = "\n\n".join(r[0] for r in rows)
    print(f"re-extracting KG from {len(rows)} stored digests "
          f"({count_tokens(text)} tokens) ...", file=sys.stderr)
    nodes, edges = extract_kg(text)
    conn.execute("DELETE FROM kg_nodes WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM kg_edges WHERE doc_id=?", (doc_id,))
    nn, ne = store_kg(conn, doc_id, None, nodes, edges)
    conn.commit()
    print(f"  knowledge graph: {nn} nodes, {ne} edges", file=sys.stderr)
    if args.out:
        n_rows, e_rows = kg_row_tuples(nodes, edges)
        _write_kg_file(n_rows, e_rows, meta, args.out)


# ---------------- Needle-in-a-haystack (unchanged scoring) ----------------
FILLER_SENTENCES = [
    "The quarterly report reviewed inventory turnover across regional warehouses.",
    "Analysts compared seasonal demand curves with prior-year shipment volumes.",
    "The committee deferred the procurement decision until the next fiscal review.",
    "Field technicians documented calibration drift in the sensor array modules.",
    "Administrative staff reconciled the expense ledgers against bank statements.",
    "The planning group scheduled maintenance windows around peak traffic hours.",
    "Researchers noted modest variance between the two measurement protocols.",
    "Procurement finalized vendor contracts for the second-half supply cycle.",
]


def build_haystack(depths):
    import random

    needles = {d: f"NEEDLE-{int(d * 100):02d}-{format(random.getrandbits(24), '06x')}"
               for d in depths}
    block = " ".join(FILLER_SENTENCES)
    parts = [block]
    body = "\n\n".join(parts)
    while count_tokens(body) < NIAH_CTX_TOKENS:
        parts.append(block)
        body = "\n\n".join(parts)
    for d, word in needles.items():
        pos = int(len(body) * d)
        line = f"NOTE: the secret codeword for this section is {word}."
        body = body[:pos] + line + "\n\n" + body[pos:]
    return body, needles


def niah(depths=(0.05, 0.25, 0.50, 0.75, 0.95)):
    body, needles = build_haystack(depths)
    print(f"Haystack ~{count_tokens(body)} tokens (server ctx must be >= this)",
          file=sys.stderr)
    user = (
        "The document below contains several secret codewords, each introduced "
        "by the phrase 'the secret codeword for this section is'. List EVERY "
        "codeword you find, one per line, in the order they appear.\n\n"
        "--- DOCUMENT ---\n" + body
    )
    response = chat(
        "You perform exact literal retrieval from documents. You never "
        "hallucinate codewords; only report strings you actually see.",
        user, max_tokens=400, temp=0.0)
    hits = 0
    for d, word in needles.items():
        ok = word.lower() in response.lower()
        hits += ok
        print(f"{int(d * 100):>7}% | {'PASS' if ok else 'FAIL'} | {word}")
    middle = [needles[d].lower() in response.lower() for d in (0.25, 0.50, 0.75)]
    print(f"overall recall: {hits}/{len(needles)}")
    print(f"attention-basin gate (25/50/75% depths): "
          f"{'PASS' if all(middle) else 'FAIL'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("pdf")
    s.add_argument("--kg-out", default=None,
                   help="'auto' (= <stem>.kg.graphml/json) or explicit filename")
    s2 = sub.add_parser("segment")
    s2.add_argument("whisper_json")
    s2.add_argument("agenda")
    s2.add_argument("--kg-out", default=None)
    sub.add_parser("niah")
    sub.add_parser("runs")
    c = sub.add_parser("compare")
    c.add_argument("--stage", default="talk_final")
    e = sub.add_parser("export-kg")
    e.add_argument("doc")  # numeric doc id, path substring, or sha256
    e.add_argument("--out", default="kg.json")  # .json, .graphml, or .dot
    r = sub.add_parser("reextract-kg")
    r.add_argument("doc")
    r.add_argument("--out", default=None)
    args = ap.parse_args()
    conn = db()
    t0 = time.time()
    if args.cmd == "summarize":
        cmd_summarize_pdf(args, conn)
    elif args.cmd == "segment":
        cmd_segment(args, conn)
    elif args.cmd == "runs":
        cmd_runs(args, conn)
    elif args.cmd == "compare":
        cmd_compare(args, conn)
    elif args.cmd == "export-kg":
        cmd_export_kg(args, conn)
    elif args.cmd == "reextract-kg":
        cmd_reextract_kg(args, conn)
    else:
        niah()
    conn.close()
    print(f"\n[elapsed {time.time() - t0:.1f}s]", file=sys.stderr)
