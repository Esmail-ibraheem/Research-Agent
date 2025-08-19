#!/usr/bin/env python3
"""
Personal Research Companion (PRC) — single-file MVP

Features
- Fetch recent arXiv papers for a query
- Simple relevance scoring + filtering
- Summarize (OpenAI if OPENAI_API_KEY is set; otherwise extractive fallback)
- Generate Markdown digest
- Maintain a JSONL knowledge base (kb.jsonl) to avoid duplicates
- Optional watch loop

Usage examples
  python prc.py search --query "graph neural networks drug discovery" --max 25 --days 14
  python prc.py digest --query "diffusion models medical imaging" --max 30 --days 21 --top 12 --out digest.md
  python prc.py watch  --query "retrieval augmented generation" --period 3600 --top 5

Requires: requests (pip install requests)
Optional: OpenAI summaries if env OPENAI_API_KEY is set.

MIT License – use at your own risk.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
import math
from collections import Counter, defaultdict


import requests


# ------------------ CONFIG ------------------
OPENAI_API_KEY = "OPENAI-KEY"   # <--- put your real key here
OPENAI_MODEL = "gpt-4o-mini"          # or "gpt-4o"

# --------------------------- Utilities ---------------------------

KB_PATH = os.environ.get("PRC_KB", "kb.jsonl")
ARXIV_API = "http://export.arxiv.org/api/query"

if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
    raise RuntimeError("OPENAI_API_KEY is not set (or invalid). Set it at the top of the file.")


def log(msg: str):
    print(f"[PRC] {msg}", flush=True)

# ---- Typewriter output ----
def typewriter_print(text: str, cps: float = 60.0):
    import sys, time
    delay = 1.0 / max(cps, 1e-6)
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if ch not in "\n\r\t ":
            time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def to_iso(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).isoformat()

def normalize_text(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", s.lower())

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def contains_any(hay: str, needles: List[str]) -> int:
    h = hay.lower()
    return sum(1 for n in needles if n.lower() in h)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()

# --------------------------- Data Models ---------------------------

@dataclass
class Paper:
    pid: str          # arXiv ID or unique URL
    title: str
    summary: str
    authors: List[str]
    link: str
    published: str    # ISO timestamp
    categories: List[str]
    score: float = 0.0

    @property
    def year(self) -> int:
        try:
            return int(self.published[:4])
        except Exception:
            return 0

# --------------------------- arXiv fetch ---------------------------

def fetch_arxiv(query: str, max_results: int = 25, days: int = 30) -> List[Paper]:
    """
    Calls arXiv Atom API and returns a list of Paper.
    """
    q = f"all:{query}"
    url = (
        f"{ARXIV_API}?search_query={quote_plus(q)}"
        f"&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    )
    log(f"Fetching arXiv: {url}")
    r = requests.get(url, headers={"User-Agent": "PRC/0.1 (+https://example)"},
                     timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    out: List[Paper] = []
    cutoff = now_utc() - dt.timedelta(days=days)

    for entry in root.findall("atom:entry", ns):
        eid = entry.findtext("atom:id", default="", namespaces=ns) or ""
        title = normalize_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = normalize_text(entry.findtext("atom:summary", default="", namespaces=ns))
        authors = [normalize_text(a.findtext("atom:name", default="", namespaces=ns))
                   for a in entry.findall("atom:author", ns)]
        link = ""
        for l in entry.findall("atom:link", ns):
            if l.attrib.get("title") == "pdf" or l.attrib.get("rel") == "alternate":
                link = l.attrib.get("href", link)
        published = entry.findtext("atom:published", default="", namespaces=ns) or ""
        cats = [c.attrib.get("term", "") for c in entry.findall("atom:category", ns)]

        # date filter
        try:
            pub_dt = dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
        except Exception:
            pub_dt = now_utc()
        if pub_dt < cutoff:
            continue

        pid = eid.split("/")[-1] or eid
        out.append(Paper(pid=pid, title=title, summary=summary,
                         authors=authors, link=link or eid,
                         published=pub_dt.isoformat(), categories=cats))
    log(f"Fetched {len(out)} recent papers")
    return out

# --------------------------- Scoring / Filtering ---------------------------

def score_paper(p: Paper, query: str) -> float:
    qtokens = tokenize(query)
    text = f"{p.title}\n{p.summary}"
    tks = tokenize(text)
    base = jaccard(tks, qtokens)
    bonus = 0.1 * contains_any(p.title, qtokens) + 0.05 * contains_any(p.summary, qtokens)
    recency_bonus = 0.02 * max(0, 2026 - p.year)  # small bias for newer
    return base + bonus + recency_bonus

def filter_and_rank(papers: List[Paper], query: str, top: int = 10, min_score: float = 0.1) -> List[Paper]:
    for p in papers:
        p.score = score_paper(p, query)
    papers.sort(key=lambda x: x.score, reverse=True)
    return [p for p in papers if p.score >= min_score][:top]

# --------------------------- Summarization ---------------------------

def summarize_with_openai(title: str, abstract: str, link: str) -> str:
    """
    Uses OpenAI Chat Completions API to summarize a paper into 4 bullets.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    prompt = (
        "You are a research assistant. Summarize the paper below "
        "into 4 short labeled bullets: Problem, Method, Results, Limitations. "
        "Keep each bullet <= 30 words.\n\n"
        f"Title: {title}\nAbstract: {abstract}\nLink: {link}\n"
    )
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You summarize academic papers accurately."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 250,
    }
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def extractive_summary(title: str, abstract: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    # score sentences by keyword overlap with title + first sentence bias
    q = tokenize(title + " " + " ".join(sentences[:1]))
    scored = []
    for i, s in enumerate(sentences):
        score = jaccard(tokenize(s), q) + (0.05 if i == 0 else 0.0)
        scored.append((score, s))
    top = [s for _, s in sorted(scored, key=lambda x: x[0], reverse=True)[:3]]
    bullets = [
        f"**Problem**: {top[0] if top else abstract[:160]}",
        f"**Method**: {top[1] if len(top) > 1 else ''}".strip(),
        f"**Results**: {top[2] if len(top) > 2 else ''}".strip(),
        f"**Limitations**: (not stated in abstract)",
    ]
    return "\n".join(f"- {b}" for b in bullets)

def summarize(p: Paper) -> str:
    try:
        return summarize_with_openai(p.title, p.summary, p.link)
    except Exception as e:
        log(f"OpenAI failed, fallback: {e}")
        return extractive_summary(p.title, p.summary)

# --------------------------- Knowledge Base ---------------------------

def kb_load_ids(path: str = KB_PATH) -> set:
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                ids.add(obj["pid"])
            except Exception:
                continue
    return ids

def kb_append(paper: Paper, summary: str, path: str = KB_PATH):
    record = {
        "pid": paper.pid,
        "title": paper.title,
        "summary": paper.summary,
        "authors": paper.authors,
        "link": paper.link,
        "published": paper.published,
        "categories": paper.categories,
        "score": paper.score,
        "digest_time": to_iso(now_utc()),
        "structured_summary": summary,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# --------------------------- Digest ---------------------------

def make_digest_md(query: str, papers: List[Paper], include_summaries: bool = True) -> str:
    lines = []
    lines.append(f"# Research Digest — {query}")
    lines.append(f"_Generated: {to_iso(now_utc())}_\n")
    for i, p in enumerate(papers, 1):
        lines.append(f"## {i}. {p.title}")
        lines.append(f"- **Score**: {p.score:.3f}")
        lines.append(f"- **Authors**: {', '.join(p.authors) if p.authors else 'N/A'}")
        lines.append(f"- **Published**: {p.published}")
        lines.append(f"- **Categories**: {', '.join(p.categories) if p.categories else '—'}")
        lines.append(f"- **Link**: {p.link}")
        if include_summaries:
            summ = summarize(p)
            lines.append("\n" + summ + "\n")
        else:
            lines.append("\n" + textwrap.shorten(p.summary, 600) + "\n")
        lines.append("---\n")
    return "\n".join(lines)


# ------------- KB loading + TF-IDF retrieval -------------

def kb_load_records(path: str = KB_PATH) -> list[dict]:
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs

def build_tfidf_index(docs: list[dict]):
    """
    Build a tiny TF-IDF index over title + structured_summary + (raw) summary.
    Returns: (vocab_idf, doc_vecs, doc_norms, texts)
    """
    texts = []
    for d in docs:
        txt = " ".join([
            d.get("title",""),
            d.get("structured_summary",""),
            d.get("summary","")
        ])
        texts.append(txt)

    # doc-term counts
    tokenized = [tokenize(t) for t in texts]
    dfs = Counter()
    for terms in tokenized:
        dfs.update(set(terms))
    N = len(tokenized)
    vocab_idf = {t: math.log((N+1) / (df+1)) + 1.0 for t, df in dfs.items()}

    # TF-IDF vectors
    doc_vecs = []
    doc_norms = []
    for terms in tokenized:
        tf = Counter(terms)
        vec = {t: (tf[t] * vocab_idf.get(t, 0.0)) for t in tf}
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        doc_vecs.append(vec)
        doc_norms.append(norm)
    return vocab_idf, doc_vecs, doc_norms, texts

def tfidf_vec(q: str, vocab_idf: dict) -> tuple[dict, float]:
    qterms = tokenize(q)
    tf = Counter(qterms)
    vec = {t: (tf[t] * vocab_idf.get(t, 0.0)) for t in tf}
    norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
    return vec, norm

def cosine_sparse(a: dict, na: float, b: dict, nb: float) -> float:
    if na == 0 or nb == 0: return 0.0
    if len(a) > len(b): a, b = b, a  # iterate smaller
    dot = 0.0
    for k, v in a.items():
        if k in b: dot += v * b[k]
    return dot / (na * nb)

def retrieve(query: str, docs: list[dict], index, k: int = 5) -> list[tuple[int, float]]:
    vocab_idf, doc_vecs, doc_norms, _ = index
    qv, qn = tfidf_vec(query, vocab_idf)
    scored = []
    for i, dv in enumerate(doc_vecs):
        score = cosine_sparse(qv, qn, dv, doc_norms[i])
        scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s in scored[:k] if s[1] > 0]

# ------------- Answer composition -------------

def answer_with_openai(question: str, context_recs: list[dict]) -> str:
    """
    Compose an answer using OpenAI, citing sources as [#].
    Falls back to a deterministic extractive answer if OpenAI fails.
    """
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        # Build compact context
        ctx_blocks = []
        for idx, r in enumerate(context_recs, start=1):
            ctx_blocks.append(
                f"[{idx}] {r.get('title','')}\n"
                f"{r.get('structured_summary','') or r.get('summary','')}\n"
                f"Link: {r.get('link','')}\n"
            )
        context = "\n\n".join(ctx_blocks)
        prompt = (
            "You are a precise research assistant. Use ONLY the provided context to answer.\n"
            "Cite with bracketed reference numbers like [1], [2]. If unknown, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "Answer concisely and accurately with citations."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        }
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log(f"OpenAI answer failed, fallback: {e}")
        # Simple extractive fallback: join top summaries
        parts = []
        for idx, r in enumerate(context_recs, start=1):
            parts.append(f"[{idx}] {r.get('title','')}\n{textwrap.shorten(r.get('structured_summary') or r.get('summary',''), 500)}")
        parts.append("\n(Answer generation unavailable; showing top sources.)")
        return "\n\n".join(parts)

def answer_with_openai_stream(question: str, context_recs: list[dict], cps: float = 60.0) -> str:
    """
    Streams an answer using OpenAI Chat Completions with stream=True.
    Also returns the full text for logging if you need it later.
    """
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        ctx_blocks = []
        for idx, r in enumerate(context_recs, start=1):
            ctx_blocks.append(
                f"[{idx}] {r.get('title','')}\n"
                f"{r.get('structured_summary','') or r.get('summary','')}\n"
                f"Link: {r.get('link','')}\n"
            )
        context = "\n\n".join(ctx_blocks)
        prompt = (
            "You are a precise research assistant. Use ONLY the provided context to answer.\n"
            "Cite with bracketed reference numbers like [1], [2]. If unknown, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "Answer concisely and accurately with citations."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 600,
            "stream": True,
        }
        with requests.post("https://api.openai.com/v1/chat/completions",
                           headers=headers, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            full = []
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                else:
                    continue
                if data.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        full.append(delta)
                        import sys, time
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        if cps > 0:
                            for ch in delta:
                                if ch not in "\n\r\t ":
                                    time.sleep(1.0/max(cps,1e-6))
                except Exception:
                    continue
        print()
        return "".join(full).strip()
    except Exception as e:
        log(f"OpenAI stream failed, fallback: {e}")
        parts = []
        for idx, r in enumerate(context_recs, start=1):
            parts.append(f"[{idx}] {r.get('title','')}\n{textwrap.shorten(r.get('structured_summary') or r.get('summary',''), 500)}")
        fallback = "\n\n".join(parts) + "\n\n(Answer generation unavailable; showing top sources.)"
        typewriter_print(fallback, cps=cps)
        return fallback


def cmd_chat(args):
    docs = kb_load_records()
    if not docs:
        log("KB is empty. Run a 'digest' first to populate kb.jsonl.")
        return
    index = build_tfidf_index(docs)
    cps = float(args.cps)
    log(f"Chat ready. {len(docs)} papers indexed. Commands: :help  :list  :open N  :sum N  :speed X  :exit")

    def list_top(q: str, k: int = 5):
        matches = retrieve(q, docs, index, k=k)
        if not matches:
            print("No matches.")
            return []
        for rank, (i, s) in enumerate(matches, start=1):
            print(f"{rank}. [{i}] {docs[i].get('title','')}  (score={s:.3f})")
        return matches

    while True:
        try:
            q = input("\nPRC> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in (":q", ":quit", ":exit"):
            print("[PRC] Chat ended.")
            break
        if q in (":h", ":help"):
            print("Commands:\n"
                  "  Any question → retrieve & answer with citations (streams)\n"
                  "  :list <query>         list top matches for a query\n"
                  "  :open <idx>           show metadata for KB index\n"
                  "  :sum <idx>            summarize a specific KB record (streams if OpenAI ok)\n"
                  "  :speed <cps>          set typing speed (chars/sec; 0 = instant)\n"
                  "  :exit                 quit")
            continue
        if q.startswith(":speed "):
            try:
                cps = float(q.split()[1]); print(f"[PRC] cps set to {cps}")
            except Exception:
                print("Usage: :speed <chars_per_second>  (e.g., :speed 80)")
            continue
        if q.startswith(":list "):
            list_top(q[6:].strip(), k=args.k)
            continue
        if q.startswith(":open "):
            try:
                i = int(q.split()[1])
                r = docs[i]
                print(json.dumps({
                    "title": r.get("title"),
                    "authors": r.get("authors"),
                    "published": r.get("published"),
                    "categories": r.get("categories"),
                    "link": r.get("link"),
                }, ensure_ascii=False, indent=2))
            except Exception:
                print("Usage: :open <idx>   (use indexes from :list)")
            continue
        if q.startswith(":sum "):
            try:
                i = int(q.split()[1])
                r = docs[i]
                fake = Paper(pid=r.get("pid",""), title=r.get("title",""),
                             summary=r.get("summary",""), authors=r.get("authors",[]),
                             link=r.get("link",""), published=r.get("published",""),
                             categories=r.get("categories",[]))
                # try to stream via OpenAI; else fallback typewriter
                try:
                    text = summarize_with_openai(fake.title, fake.summary, fake.link)
                    typewriter_print(text, cps=cps)
                except Exception:
                    typewriter_print(extractive_summary(fake.title, fake.summary), cps=cps)
            except Exception:
                print("Usage: :sum <idx>")
            continue

        # Default: retrieve → stream an answer
        matches = retrieve(q, docs, index, k=args.k)
        if not matches:
            print("No relevant items found. Try a different query or :list <query>.")
            continue
        top_recs = [docs[i] for (i, _s) in matches]
        _full = answer_with_openai_stream(q, top_recs, cps=cps)
        # Show sources (non-stream)
        print("\nSources:")
        for ref_num, (i, s) in enumerate(matches, start=1):
            r = docs[i]
            print(f"  [{ref_num}] {r.get('title','')}  — {r.get('link','')}  (score={s:.3f})")

# --------------------------- CLI Commands ---------------------------

def cmd_search(args):
    papers = fetch_arxiv(args.query, args.max, args.days)
    ranked = filter_and_rank(papers, args.query, top=args.top, min_score=args.min_score)
    for i, p in enumerate(ranked, 1):
        print(f"{i:2d}. {p.title}  ({p.published})")
        print(f"    score={p.score:.3f}  link={p.link}")
        print(f"    authors={', '.join(p.authors)}")
        print()

def cmd_digest(args):
    seen = kb_load_ids()
    papers = fetch_arxiv(args.query, args.max, args.days)
    ranked = filter_and_rank(papers, args.query, top=args.top, min_score=args.min_score)

    # Generate summaries and update KB
    for p in ranked:
        s = summarize(p)
        kb_append(p, s)

    md = make_digest_md(args.query, ranked, include_summaries=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    log(f"Digest written to {args.out}")

def cmd_watch(args):
    period = max(300, args.period)  # >=5min
    log(f"Entering watch mode (every {period}s). Ctrl+C to stop.")
    while True:
        try:
            cmd_digest(args)  # reuse same args to regenerate digest & append KB
        except Exception as e:
            log(f"watch iteration error: {e}")
        time.sleep(period)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Personal Research Companion (PRC)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Fetch and print ranked papers")
    s.add_argument("--query", required=True)
    s.add_argument("--max", type=int, default=30)
    s.add_argument("--days", type=int, default=30)
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--min-score", type=float, default=0.1)
    s.set_defaults(func=cmd_search)

    d = sub.add_parser("digest", help="Create a Markdown digest and update KB")
    d.add_argument("--query", required=True)
    d.add_argument("--max", type=int, default=40)
    d.add_argument("--days", type=int, default=30)
    d.add_argument("--top", type=int, default=12)
    d.add_argument("--min-score", type=float, default=0.1)
    d.add_argument("--out", default="digest.md")
    d.set_defaults(func=cmd_digest)

    w = sub.add_parser("watch", help="Regenerate digest on an interval")
    w.add_argument("--query", required=True)
    w.add_argument("--max", type=int, default=40)
    w.add_argument("--days", type=int, default=14)
    w.add_argument("--top", type=int, default=10)
    w.add_argument("--min-score", type=float, default=0.12)
    w.add_argument("--out", default="digest.md")
    w.add_argument("--period", type=int, default=3600, help="Seconds between runs")
    w.set_defaults(func=cmd_watch)

    c = sub.add_parser("chat", help="Interactive terminal chat over your KB (RAG)")
    c.add_argument("--k", type=int, default=5, help="Top-k documents to retrieve per question")
    c.add_argument("--cps", type=float, default=60.0, help="Typing speed (chars/sec, 0 = instant)")
    c.set_defaults(func=cmd_chat)



    return p

# --------------------------- Main ---------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
