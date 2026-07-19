"""
Streamlit UI — RAG Query Mapping Module (REHAU RAUVISIO)

Run: streamlit run app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import EVAL_RESULTS, CHROMA_DIR, GEN_MODEL
from src.qrels import load_qrels
from src.retriever import retrieve_full, retrieve_full_rrf
from src.strategies.nlp_strategy import nlp_map
from src.strategies.embedding_strategy import embedding_map
from src.strategies.llm_strategy import llm_map

st.set_page_config(page_title="REHAU Query Mapping", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Checking Ollama...")
def llm_available() -> bool:
    try:
        import ollama
        ollama.show(GEN_MODEL)
        return True
    except Exception:
        return False


@st.cache_data
def get_qrels() -> dict:
    return load_qrels()


@st.cache_data
def get_eval_results():
    if os.path.exists(EVAL_RESULTS):
        import json
        with open(EVAL_RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return None


def run_strategy(name, query, llm_ok, method="few_shot"):
    if name == "Baseline":
        return {"original_query": query, "final_query": query, "strategy": "Baseline"}
    if name == "NLP-based":
        return nlp_map(query)
    if name == "Embedding-based":
        return embedding_map(query)
    if name == "LLM-based":
        return llm_map(query, llm_available=llm_ok, method=method)
    return {"original_query": query, "final_query": query}


# Curated demos that hit terminology (flat_map / product phrases) so NLP
# expansions are visible — the first eval queries often abstain by design.
DEMO_QUERIES = [
    "What's the fire rating for these metal look panels?",
    "How do I put these shiny acrylic panels up?",
    "For my crystal glass product, how do I cut a beveled edge on this clear material without breaking it?",
    "How do I safely take my shade matte boards out of the box without messing them up?",
    "How much weight can these floating shelves hold?",
    "What's the fire safety rating for this noir matte material?",
    "For my wood-grain product, how should I store these laminate sheets before putting them up?",
    "What's the fire safety rating of this wood-grain material when attached to drywall?",
]


st.title("RAG Query Mapping Module")
st.caption("ABA SS2026 · Bridging vocabulary mismatch in REHAU RAUVISIO documentation retrieval")

llm_ok = llm_available()
qrels = get_qrels()
examples = DEMO_QUERIES

c1, c2, c3 = st.columns(3)
c1.metric("Ground-truth queries", len(qrels))
c2.metric("LLM (Ollama)", f"Yes ({GEN_MODEL})" if llm_ok else "No")
c3.metric("Vector index", "Ready" if os.path.exists(CHROMA_DIR) else "Not built")

if not os.path.exists(CHROMA_DIR):
    st.error("ChromaDB not found. Run `python scripts/index.py` first.")
    st.stop()

st.divider()
tab_demo, tab_compare, tab_results = st.tabs(
    ["Live Demo", "Compare All Strategies", "Evaluation Results"])

with tab_demo:
    col_in, col_out = st.columns([1, 1], gap="large")
    with col_in:
        st.subheader("Input")
        example = st.selectbox(
            "Example (chosen to show NLP term expansion):",
            ["(type your own)"] + examples,
        )
        query = st.text_area(
            "Your query:", height=90,
            value="" if example == "(type your own)" else example,
            placeholder="Ask in casual everyday language...",
        )
        strategy = st.radio("Strategy:",
                            ["Baseline", "NLP-based", "Embedding-based", "LLM-based"],
                            horizontal=True)
        method = "few_shot"
        if strategy == "LLM-based":
            method = st.selectbox("LLM method:",
                                  ["few_shot", "zero_shot", "cot", "multi_query"])
        top_k = st.slider("Top-K documents:", 1, 20, 5)
        go = st.button("Run", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Step-by-step output")
        if go and query.strip():
            gold = set(qrels[query]["gold_chunk_ids"]) if query in qrels else set()
            with st.expander("Step 1 — Original query", expanded=True):
                st.info(f'"{query}"')
            with st.expander("Step 2 — Mapping module", expanded=True):
                with st.spinner("Mapping..."):
                    result = run_strategy(strategy, query, llm_ok, method)
                mapped = result.get("final_query", query)
                if strategy == "NLP-based":
                    st.markdown("**Term mappings** *(embedding-gated; RRF with original)*")
                    if result.get("abstained"):
                        st.caption("Abstained — retrieving original query only.")
                    terms = result.get("formal_terms") or []
                    if terms:
                        for t in terms:
                            label = t.get("source", "term")
                            sim = t.get("sim")
                            sim_s = f" · sim {sim:.2f}" if isinstance(sim, (int, float)) else ""
                            if t.get("from") and t["from"] != "(matched)":
                                st.markdown(
                                    f"- `{t['from']}` → **{t['to']}** *{label}*{sim_s}"
                                )
                            else:
                                st.markdown(f"- **{t['to']}** *{label}*{sim_s}")
                    elif not result.get("abstained"):
                        st.caption("No terminology terms to append.")
                    if result.get("appended_terms"):
                        st.caption(
                            "Appended: " + " | ".join(result["appended_terms"])
                        )
                elif strategy == "Embedding-based":
                    st.markdown(
                        "**Product-line semantic expansion** *(embedding lookup; RRF with original)*"
                    )
                    dbg = result.get("debug", {})
                    if result.get("abstained"):
                        kept = dbg.get("kept") or {}
                        st.caption(
                            f"Abstained — {result.get('method_used', 'passthrough')}. "
                            f"Gate ≥ {dbg.get('gate_min_sim', '—')}"
                            + (f" · best was {kept}" if kept else "")
                        )
                    terms = result.get("formal_terms") or []
                    if terms:
                        for t in terms:
                            st.markdown(
                                f"- `{t.get('from', '')}` → **{t.get('to', '')}** "
                                f"*{t.get('source', '')}* · sim {t.get('sim', 0):.2f}"
                            )
                    if result.get("appended_terms"):
                        st.caption(
                            "Appended: " + " | ".join(result["appended_terms"])
                        )
                elif strategy == "LLM-based" and method == "multi_query":
                    for i, v in enumerate(result.get("query_variants", []), 1):
                        st.markdown(f"{i}. {v}")
                if mapped.strip() != query.strip():
                    st.caption("Expanded query")
                    st.success(mapped[:500])
                else:
                    st.caption("Expanded query (unchanged)")
                    st.text(mapped[:500])
            with st.expander("Step 3 — Retrieved documents", expanded=True):
                with st.spinner("Retrieving..."):
                    if strategy == "LLM-based" and method == "multi_query":
                        docs, seen = [], set()
                        for v in result.get("query_variants", [query]):
                            for d in retrieve_full(v, top_k=top_k):
                                if d["chunk_id"] not in seen:
                                    docs.append(d)
                                    seen.add(d["chunk_id"])
                        docs = docs[:top_k]
                    elif strategy in ("NLP-based", "Embedding-based"):
                        docs = retrieve_full_rrf(
                            result.get("retrieval_queries") or [query],
                            top_k=top_k,
                        )
                    else:
                        docs = retrieve_full(mapped, top_k=top_k)
                if gold:
                    if any(d["chunk_id"] in gold for d in docs):
                        st.success("Gold chunk found ✓")
                    else:
                        st.error("Gold chunk NOT found ✗")
                if not docs:
                    st.warning("No documents retrieved.")
                for i, doc in enumerate(docs, 1):
                    mark = "✓ " if doc["chunk_id"] in gold else ""
                    with st.container(border=True):
                        st.markdown(
                            f"{mark}**{i}.** `{doc['chunk_id']}` — "
                            f"score {doc['score']} · {doc.get('product', '')} · "
                            f"{doc.get('source', '')}"
                        )
                        st.text(doc["document"][:800])
            if query in qrels:
                with st.expander("Reference answer"):
                    st.write(qrels[query]["formal_query"])
                    st.write(qrels[query]["answer"])

with tab_compare:
    ex = st.selectbox("Query:", examples or ["(none)"])
    custom = st.text_input("Or type your own:")
    cmp_q = custom.strip() or ex
    if st.button("Compare all", type="primary"):
        gold = set(qrels[cmp_q]["gold_chunk_ids"]) if cmp_q in qrels else set()
        for strat in ["Baseline", "NLP-based", "Embedding-based", "LLM-based"]:
            st.markdown(f"**{strat}**")
            res = run_strategy(strat, cmp_q, llm_ok)
            if strat in ("NLP-based", "Embedding-based"):
                docs = retrieve_full_rrf(
                    res.get("retrieval_queries") or [cmp_q],
                    top_k=3,
                )
            else:
                docs = retrieve_full(res.get("final_query", cmp_q), top_k=3)
            st.text(res.get("final_query", cmp_q)[:200])
            if any(d["chunk_id"] in gold for d in docs):
                st.markdown("✓ GT found")
            else:
                st.markdown("✗ GT missing")

with tab_results:
    eval_res = get_eval_results()
    if eval_res:
        rows = [{"Strategy": s, **{f"R@{k}": m["recall"].get(f"@{k}", "-")
                                     for k in (1, 3, 5)},
                 "MRR": m["mrr"], "nDCG@5": m["ndcg@5"]}
                for s, m in eval_res.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No results yet. Run `python scripts/evaluate.py --no-llm`.")
