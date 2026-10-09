"""Thin Streamlit page for DevSearch: a query box, a baseline/rerank/compare switch and the ranked lists side by side.

Everything that can be checked without Streamlit is a pure function here; `main()` is the only place that imports
`streamlit` (and `call_api` the only place that imports `httpx`), so importing this module for tests or tooling
stays cheap. Run it with `streamlit run src/devsearch/serving/ui.py --server.address 127.0.0.1 --server.port 8501
--browser.gatherUsageStats false` (loopback only, no telemetry). The page loads no model: it calls the local API
(`DEVSEARCH_API_URL`, default http://127.0.0.1:8000) and shows what comes back.

Result text, titles and locations are untrusted data (docs/04). They reach the page only through `st.text` and
`st.code`, never as markdown or HTML, and a source link is drawn only for an https URL on a configured upstream host.
"""

import math
import os
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

API_URL_ENV = "DEVSEARCH_API_URL"
SOURCE_HOSTS_ENV = "DEVSEARCH_SOURCE_HOSTS"
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_SOURCE_HOSTS = ("pytorch.org", "github.com", "huggingface.co")    # the proposed corpora's upstreams (docs/04 references)
REQUEST_TIMEOUT_S = 10.0                                                    # short: the API answers 429/504 on its own
MODES = ("baseline", "rerank", "compare")
LOGIT_LABEL = "uncalibrated relevance score (not a probability, not comparable to BM25)"
LIMITATION_NOTE = "Ranked evidence only: a score orders passages; it does not say that an answer is correct, and no score proves that no answer exists."


# -- configuration and request --------------------------------------------------------------------------
def api_base_url(environ: Mapping[str, str] | None = None) -> str:
    """The API root from `DEVSEARCH_API_URL`, without a trailing slash; http(s) with a host only."""
    value = ((os.environ if environ is None else environ).get(API_URL_ENV) or DEFAULT_API_URL).strip().rstrip("/")
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{API_URL_ENV} must be an http(s) URL with a host, got {value!r}")
    return value


def source_hosts(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Hosts a source link may point to: `DEVSEARCH_SOURCE_HOSTS` (comma-separated; empty means no links) or the defaults."""
    raw = (os.environ if environ is None else environ).get(SOURCE_HOSTS_ENV)
    if raw is None:
        return DEFAULT_SOURCE_HOSTS
    return tuple(h.strip().lower() for h in raw.split(",") if h.strip())


def parse_source_filter(text: str) -> list[str]:
    """Source ids from a comma/space separated box. Nothing is deduplicated or corrected: the API judges them."""
    return [part for part in text.replace(",", " ").split() if part]


def build_payload(
    query: str, mode: str = "baseline", sources: Iterable[str] = (), *,
    candidate_k: int | None = None, top_k: int | None = None, versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The `POST /search` body. Unset options are omitted so the API applies its configured defaults."""
    payload: dict[str, Any] = {"query": query, "mode": mode}
    if sources := list(sources):
        payload["sources"] = sources
    if versions:
        payload["versions"] = dict(versions)
    for name, value in (("candidate_k", candidate_k), ("top_k", top_k)):
        if value is not None:
            payload[name] = value
    return payload


# -- display helpers -------------------------------------------------------------------------------------
def _number(value: Any, digits: int) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return f"{value:.{digits}f}"


def format_logit(value: Any) -> str:
    """A reranker logit, always labelled: it is not a probability and cannot be compared with a BM25 score."""
    text = _number(value, 3)
    return "not reranked" if text is None else f"{text} - {LOGIT_LABEL}"


def format_bm25(value: Any) -> str:
    return _number(value, 3) or "n/a"


def format_location(location: Any) -> str:
    """`source_location` (file lines or notebook cell character spans) as one line of plain text."""
    parts = []
    for item in location if isinstance(location, Sequence) and not isinstance(location, str) else ():
        if not isinstance(item, Mapping):
            continue
        if "cell_id" in item:
            parts.append(f"{item.get('path')} cell {item.get('cell_id')} chars {item.get('start_char')}-{item.get('end_char')}")
        elif item.get("start_line") == item.get("end_line"):
            parts.append(f"{item.get('path')} line {item.get('start_line')}")
        else:
            parts.append(f"{item.get('path')} lines {item.get('start_line')}-{item.get('end_line')}")
    return "; ".join(parts) or "location unavailable"


def safe_link(url: Any, allowed_hosts: Collection[str] | None = None) -> str | None:
    """`url` if it is a well-formed https URL (no credentials, no whitespace) on an allowed host or its subdomain, else None.
    `allowed_hosts=None` means the configured hosts; an empty collection allows nothing."""
    if not isinstance(url, str) or not url or len(url) > 2048 or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        return None
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return None
    if parts.scheme != "https" or not host or "@" in parts.netloc:
        return None
    hosts = source_hosts() if allowed_hosts is None else [h.lower() for h in allowed_hosts]
    return url if any(host == h or host.endswith("." + h) for h in hosts) else None


def result_rows(results: Iterable[Mapping[str, Any]] | None, *, allowed_hosts: Collection[str] | None = None) -> list[dict[str, Any]]:
    """Display rows for a result list: the fields the page shows, with numbers and locations already formatted."""
    return [{
        "chunk_id": r.get("chunk_id"), "title": r.get("title") or "(untitled)", "section": r.get("section") or "",
        "source_id": r.get("source_id"), "revision": r.get("revision"), "source_url": r.get("source_url"),
        "link": safe_link(r.get("source_url"), allowed_hosts), "location": format_location(r.get("source_location")),
        "baseline_rank": r.get("baseline_rank"), "reranked_rank": r.get("reranked_rank"),
        "bm25_score": r.get("bm25_score"), "rerank_logit": r.get("rerank_logit"),
        "bm25_text": format_bm25(r.get("bm25_score")), "logit_text": format_logit(r.get("rerank_logit")),
        "kind": "code excerpt" if r.get("code_fragment") else "passage", "text": r.get("text") or "",
    } for r in results or ()]


def row_summary(row: Mapping[str, Any], list_name: str) -> str:
    """Plain-text header block for one result in the `baseline` or `reranked` list."""
    rank = row["reranked_rank"] if list_name == "reranked" else row["baseline_rank"]
    lines = [f"#{rank}  {row['title']}"]
    if row["section"]:
        lines.append(f"Section: {row['section']}")
    lines.append(f"Source: {row['source_id']} @ {row['revision']} - {row['location']}")
    lines.append(f"BM25 rank {row['baseline_rank']} (BM25 score {row['bm25_text']})")
    if list_name == "reranked":
        lines.append(f"Rerank score: {row['logit_text']}")
    lines.append(f"Shown as: {row['kind']}")
    return "\n".join(lines)


def format_timings(timings: Mapping[str, Any] | None) -> str:
    timings = timings or {}
    parts = []
    for name in ("retrieval", "rerank", "total"):
        text = _number(timings.get(name), 1)
        parts.append(f"{name} {text} ms" if text else f"{name} not run" if name == "rerank" else f"{name} n/a")
    return " | ".join(parts)


def describe_response(document: Mapping[str, Any], *, allowed_hosts: Collection[str] | None = None) -> dict[str, Any]:
    """Everything the page needs from a search response, with no Streamlit involved.

    `layout` is `compare` only when both lists exist; a degraded response keeps just the correctly labelled baseline.
    `banner.text` is static wording (safe for markdown); server-provided text stays in `warnings` for plain-text display.
    """
    baseline = result_rows(document.get("baseline_results"), allowed_hosts=allowed_hosts)
    reranked = result_rows(document.get("reranked_results"), allowed_hosts=allowed_hosts)
    status = document.get("status")
    banner: dict[str, Any] = {"level": None, "text": ""}
    if status == "degraded":
        banner = {"level": "warning", "text": "Reranking was not applied, so these are the BM25 baseline results. No partial reranking is shown."}
    elif status == "empty":
        banner = {"level": "info", "text": "No passages matched this query with the current filters. That means no lexical match, not that no answer exists."}
    layout = "compare" if baseline and reranked else "reranked" if reranked else "baseline" if baseline else "none"
    return {
        "status": status, "mode_requested": document.get("mode_requested"), "mode_effective": document.get("mode_effective"),
        "layout": layout, "banner": banner, "note": LIMITATION_NOTE,
        "baseline_title": "BM25 baseline" + (" (reranking not applied)" if status == "degraded" else " (lexical candidates)"),
        "reranked_title": "Reranked (same candidates)",
        "baseline": baseline, "reranked": reranked,
        "counts": {"candidates": document.get("candidate_count"), "baseline": len(baseline), "reranked": len(reranked)},
        "warnings": [f"{w.get('code')}: {w.get('message')}" for w in document.get("warnings") or [] if isinstance(w, Mapping)],
        "timings": format_timings(document.get("timings_ms")),
        "identity": {k: document.get(k) for k in ("request_id", "release_id", "corpus_id", "index_id", "model_id", "candidate_set_id")},
    }


_ERROR_TITLES = {
    422: ("The search request was rejected", "Fix the query or filters and try again."),
    429: ("The search service is busy", "Only one search runs at a time and nothing is queued; try again in a moment."),
    503: ("Search is unavailable", "The service is up but its index (or a required part of the release) cannot be used."),
    504: ("The search ran out of time", "The service may still be finishing this search and stays busy until it stops; try again shortly."),
    500: ("The search service had an internal error", "Details are in the server's local log."),
}


def describe_error(status: int | None, body: Any) -> dict[str, Any]:
    """A readable error from an API error body (or from a response that is not one). `title` and `hint` are static
    wording; `detail` carries server text and must be shown as plain text."""
    error = body.get("error") if isinstance(body, Mapping) else None
    title, hint = _ERROR_TITLES.get(status, (f"Unexpected response from the API (HTTP {status})", ""))
    if not isinstance(error, Mapping):
        return {"status": status, "code": None, "title": title, "detail": "The response was not a DevSearch error body.",
                "hint": hint, "retryable": None, "request_id": None}
    return {"status": status, "code": error.get("code"), "title": title, "detail": str(error.get("message") or ""), "hint": hint,
            "retryable": error.get("retryable"), "request_id": body.get("request_id")}


def call_api(
    payload: Mapping[str, Any], *, base_url: str | None = None, timeout: float = REQUEST_TIMEOUT_S, client: Any = None,
) -> dict[str, Any]:
    """POST the search. Returns `{"ok": True, "document": ...}` or `{"ok": False, ...describe_error fields}`; it never raises
    for a network or protocol failure. Proxies from the environment are ignored: the API is local."""
    import httpx

    root = base_url or api_base_url()
    owned = client is None
    client = client or httpx.Client(timeout=timeout, trust_env=False)
    try:
        response = client.post(f"{root}/search", json=dict(payload), timeout=timeout)
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code == 200 and isinstance(body, Mapping):
            return {"ok": True, "document": body}
        return {"ok": False, **describe_error(response.status_code, body)}
    except httpx.TimeoutException:
        return {"ok": False, "status": None, "code": "client_timeout", "title": "The API did not answer in time",
                "detail": f"No answer from {root} within {timeout:g} s.", "retryable": True, "request_id": None,
                "hint": "It may still be finishing an earlier search and stays busy until that ends; try again shortly."}
    except httpx.HTTPError as error:
        return {"ok": False, "status": None, "code": "unreachable", "title": "Cannot reach the DevSearch API",
                "detail": f"{type(error).__name__} while contacting {root}.", "retryable": True, "request_id": None,
                "hint": f"Start the API service, or set {API_URL_ENV} to its address."}
    finally:
        if owned:
            client.close()


# -- the page --------------------------------------------------------------------------------------------
def _render_list(st: Any, heading: str, rows: Sequence[Mapping[str, Any]], list_name: str) -> None:
    st.subheader(heading)
    if not rows:
        st.text("No results.")
    for position, row in enumerate(rows):
        with st.container(border=True):
            st.text(row_summary(row, list_name))
            st.code(row["text"], language="text", wrap_lines=True)         # plain text, never markdown or HTML
            if row["link"]:
                st.link_button("Open source page", row["link"], key=f"link-{list_name}-{position}")
            elif row["source_url"]:
                st.text(f"Source URL (not linked): {row['source_url']}")


def main() -> None:
    """The Streamlit script body. Streamlit is imported here and nowhere else."""
    import streamlit as st

    st.set_page_config(page_title="DevSearch", layout="wide")
    st.title("DevSearch")
    st.caption("BM25 candidates, optionally reranked by a cross-encoder. This page only calls the local API.")
    with st.form("search"):
        query = st.text_input("Query")
        mode = st.radio("Mode", MODES, horizontal=True, index=MODES.index("compare"))
        sources = st.text_input("Source filter (optional, comma-separated source ids)")
        submitted = st.form_submit_button("Search")
    if not submitted:
        return
    try:
        outcome = call_api(build_payload(query, mode, parse_source_filter(sources)))
    except ValueError as error:                                             # a malformed DEVSEARCH_API_URL
        st.error("The API address is not valid.")
        st.text(str(error))
        return
    if not outcome["ok"]:
        st.error(outcome["title"])
        if outcome["detail"]:
            st.text(outcome["detail"])
        if outcome["hint"]:
            st.caption(outcome["hint"])
        return
    view = describe_response(outcome["document"])
    if view["banner"]["level"] == "warning":
        st.warning(view["banner"]["text"])
    elif view["banner"]["level"] == "info":
        st.info(view["banner"]["text"])
    for warning in view["warnings"]:
        st.text(warning)
    counts = view["counts"]
    st.text(f"Mode requested: {view['mode_requested']} | shown: {view['mode_effective']} | candidates: {counts['candidates']} | "
            f"timings: {view['timings']}")
    st.caption(view["note"])
    if view["layout"] == "compare":
        left, right = st.columns(2)
        with left:
            _render_list(st, view["baseline_title"], view["baseline"], "baseline")
        with right:
            _render_list(st, view["reranked_title"], view["reranked"], "reranked")
    elif view["layout"] == "reranked":
        _render_list(st, view["reranked_title"], view["reranked"], "reranked")
    elif view["layout"] == "baseline":
        _render_list(st, view["baseline_title"], view["baseline"], "baseline")
    st.caption("Reranker scores are " + LOGIT_LABEL + ".")
    with st.expander("Identities"):
        st.text("\n".join(f"{k}: {v}" for k, v in view["identity"].items()))


if __name__ == "__main__":                                                  # `streamlit run` executes the file as __main__
    main()
