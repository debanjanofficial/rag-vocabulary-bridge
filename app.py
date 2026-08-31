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

from config import (
    ANSWER_MODEL,
    EVAL_RESULTS,
    CHROMA_DIR,
    GEN_MODEL,
    RERANK_TOP_N,
)
from src.answer_pipeline import retrieve_candidates, run_answer_pipeline
from src.qrels import load_qrels
from src.reranker import rerank_documents
from src.router.router import route_query
from src.strategies.nlp_strategy import nlp_map
from src.strategies.embedding_strategy import embedding_map
from src.strategies.llm_strategy import llm_map


st.set_page_config(page_title="REHAU Query Mapping", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def warm_embed_model():
    from src.embeddings import get_model
    return get_model()


@st.cache_resource(show_spinner="Loading cross-encoder reranker...")
def warm_rerank_model():
    from src.reranker import get_reranker
    return get_reranker()


@st.cache_resource(show_spinner="Checking Ollama models...")
def ollama_status() -> dict:
    """Which Ollama models are available (mapping 8b vs answer 0.6b)."""
    try:
        import ollama
    except Exception:
        return {"mapping": False, "answer": False}
    mapping = answer = False
    try:
        ollama.show(GEN_MODEL)
        mapping = True
    except Exception:
        pass
    try:
        ollama.show(ANSWER_MODEL)
        answer = True
    except Exception:
        pass
    return {"mapping": mapping, "answer": answer}


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


def run_strategy(name, query, llm_ok):
    if name in ("Adaptive (Auto-Route)", "Adaptive"):
        return route_query(query, llm_available=llm_ok).to_dict()
    if name == "Baseline":
        return {"original_query": query, "final_query": query, "strategy": "Baseline"}
    if name == "NLP-based":
        return nlp_map(query)
    if name == "Embedding-based":
        return embedding_map(query)
    if name == "LLM-based":
        return llm_map(query, llm_available=llm_ok)
    return {"original_query": query, "final_query": query}


DEMO_QUERIES = [
    # From eval_dataset — have gold chunks + reference answers in the UI
    "How does this metal-look material burn?",
    "What's the fire safety rating for this noir matte material?",
    "For my wood-grain product, how should I store these laminate sheets before putting them up?",
    "How do I drill holes in my crystal glass panels without breaking them?",
    "How do I make sure the edges look good when cutting my wood-grain panels?",
    "For my floating shelf product, how far can the MDF strip stick out from the last stud if the shelf is between 22 and 45 inches long?",
    "What's the allowed length variation for these shade matte parts?",
    "What do I need to do to keep the metal-look product safe before and after taking it out of the box?",
    "What's the best way to store my crystal glass boards so they don't warp?",
    "How do you attach my noir matte panels together?",
    # Extra mapping demo (also in eval set)
    "How do I put these shiny acrylic panels up?",
]


st.title("RAG Query Mapping Module")
st.caption("ABA SS2026 · Bridging vocabulary mismatch in REHAU RAUVISIO documentation retrieval")

# Warm heavy models once per session (cached).
warm_embed_model()
warm_rerank_model()
ollama = ollama_status()
llm_ok = ollama["mapping"]
answer_ok = ollama["answer"]
qrels = get_qrels()
examples = DEMO_QUERIES

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ground-truth queries", len(qrels))
c2.metric("Mapping LLM", f"{GEN_MODEL}" if llm_ok else "No")
c3.metric("Answer LLM", f"{ANSWER_MODEL}" if answer_ok else "No")
c4.metric("Vector index", "Ready" if os.path.exists(CHROMA_DIR) else "Not built")

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
            "Example query:",
            ["(type your own)"] + examples,
        )
        query = st.text_area(
            "Your query:", height=90,
            value="" if example == "(type your own)" else example,
            placeholder="Ask in casual everyday language...",
        )
        strategy = st.radio(
            "Strategy:",
            ["Adaptive (Auto-Route)", "Baseline", "NLP-based", "Embedding-based", "LLM-based"],
            horizontal=True,
        )
        do_generate = st.checkbox(
            "Generate answer",
            value=True,
            help="Off = map + retrieve + rerank only (faster).",
        )
        answer_style = st.radio(
            "Answer length:",
            ["Short", "Detailed"],
            horizontal=True,
            disabled=not do_generate,
            help="Detailed uses more context; simple facts stay short.",
        )

        detail = answer_style == "Detailed"
        go = st.button("Run", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Step-by-step output")
        if go and query.strip():
            gold = set(qrels[query]["gold_chunk_ids"]) if query in qrels else set()
            with st.expander("Step 1 — Original query", expanded=True):
                st.info(f'"{query}"')
            with st.expander("Step 2 — Mapping module", expanded=True):
                with st.spinner("Mapping..."):
                    result = run_strategy(strategy, query, llm_ok)
                mapped = result.get("final_query", query)
                if strategy in ("Adaptive (Auto-Route)", "Adaptive"):
                    dec = result.get("decision", {})
                    act = result.get("action", "passthrough")
                    fv = result.get("features", {})
                    action_badges = {
                        "passthrough": "🔵 **Action: PASSTHROUGH** (Exact Technical Keywords Detected)",
                        "expand_terminology": "🟢 **Action: EXPAND_TERMINOLOGY** (In-Place Knowledge Graph Injection)",
                        "filter_metadata": "🟣 **Action: FILTER_METADATA** (Product Constraint + Intent Reranking)",
                        "expand_semantic": "🟠 **Action: EXPAND_SEMANTIC** (Intent Query Reformulation)",
                        "clarify": "🔴 **Action: CLARIFY** (Ambiguity Detected / Abstain & Clarify)",
                    }
                    st.markdown(action_badges.get(act, f"**Action:** `{act}`"))
                    st.caption(
                        f"**Confidence:** {dec.get('confidence', 1.0):.2f} · **Rationale:** {dec.get('reason', '')}"
                    )
                    st.markdown("**Routing Feature Signatures:**")
                    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
                    f_col1.metric("S_term (Gap)", f"{fv.get('s_term', 0.0):.2f}")
                    f_col2.metric("C_prod (Conf)", f"{fv.get('c_prod', 0.0):.2f}")
                    f_col3.metric("J_agree (Overlap)", f"{fv.get('j_agree', 0.0):.2f}")
                    f_col4.metric("H_dense (Entropy)", f"{fv.get('h_dense', 0.0):.2f}")
                    f_col5.metric("R_drift (Risk)", f"{fv.get('r_drift', 0.0):.2f}")

                    if result.get("clarification_prompt"):
                        st.warning(
                            f"⚠️ **Targeted Clarification Question:**\n\n{result['clarification_prompt']}"
                        )
                    if result.get("filters"):
                        st.caption(f"Applied Filters: `{result['filters']}`")
                elif strategy == "NLP-based":
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
                elif strategy == "LLM-based":
                    st.markdown(
                        "**Self-Query** *(hard product filter + intent-aware guide/spec rerank)*"
                    )
                    dbg = result.get("debug", {}) or {}
                    if result.get("abstained"):
                        st.caption(
                            f"Abstained — {result.get('method_used', 'passthrough')}. "
                            "No confident product filter (baseline retrieval)."
                        )
                    else:
                        filters = result.get("filters") or {}
                        parts = []
                        if filters.get("product"):
                            parts.append(f"product=`{filters['product']}`")
                        if filters.get("doc_type"):
                            parts.append(f"doc_type=`{filters['doc_type']}`")
                        prefer = result.get("prefer_source")
                        if prefer:
                            parts.append(f"prefer=`{prefer}`")
                        st.caption("Filters: " + ", ".join(parts))
                    st.markdown(
                        f"**Intent:** `{result.get('intent', 'other')}`  \n"
                        f"**Semantic query:** {result.get('semantic_query', mapped)}"
                    )
                    if dbg.get("raw_llm"):
                        with st.expander("LLM JSON (raw)"):
                            st.json(dbg["raw_llm"])
                if mapped.strip() != query.strip() and strategy not in ("LLM-based", "Adaptive (Auto-Route)", "Adaptive"):
                    st.caption("Expanded query")
                    st.success(mapped[:500])
                else:
                    st.caption(
                        "Expanded query (unchanged)"
                        if strategy not in ("LLM-based", "Adaptive (Auto-Route)", "Adaptive")
                        else "Routed query payload"
                    )
                    st.text(mapped[:500])


            with st.expander(
                "Step 3 — Retrieve + semantic rerank",
                expanded=True,
            ):
                with st.spinner("Retrieving and reranking..."):
                    pipeline = run_answer_pipeline(
                        strategy,
                        query,
                        result,
                        llm_available=answer_ok,
                        generate=do_generate,
                        detail=detail,
                    )
                docs = pipeline["reranked"]
                n_cand = len(pipeline["candidates"])
                st.caption(f"Candidates: {n_cand} → reranked: {len(docs)}")
                if gold:
                    if any(d["chunk_id"] in gold for d in docs):
                        st.success("Gold chunk found in reranked set ✓")
                    else:
                        st.error("Gold chunk NOT found in reranked set ✗")
                if not docs:
                    st.warning("No documents retrieved.")
                for i, doc in enumerate(docs, 1):
                    mark = "✓ " if doc["chunk_id"] in gold else ""
                    rr = doc.get("rerank_score")
                    rr_s = f" · rerank {rr:.3f}" if isinstance(rr, (int, float)) else ""
                    with st.container(border=True):
                        st.markdown(
                            f"{mark}**{i}.** `{doc['chunk_id']}` — "
                            f"emb {doc.get('score', '—')}{rr_s} · "
                            f"{doc.get('product', '')} · {doc.get('source', '')}"
                        )
                        st.text(doc["document"][:800])

            with st.expander("Step 4 — Generated answer", expanded=True):
                gen = pipeline["generation"]
                if gen.get("error") == "skipped":
                    st.info("Generation skipped — enable the checkbox to run it.")
                elif gen.get("error") == "llm_unavailable":
                    st.warning(gen["answer"])
                elif gen.get("abstained"):
                    st.warning(gen["answer"])
                else:
                    st.success(gen["answer"])

            if query in qrels:
                with st.expander("Reference answer"):
                    st.write(qrels[query]["formal_query"])
                    st.write(qrels[query]["answer"])

with tab_compare:
    ex = st.selectbox("Query:", examples or ["(none)"])
    custom = st.text_input("Or type your own:")
    cmp_q = custom.strip() or ex
    gen_answers = st.checkbox(
        "Also generate answers", value=False,
    )
    if st.button("Compare all", type="primary"):
        gold = set(qrels[cmp_q]["gold_chunk_ids"]) if cmp_q in qrels else set()
        for strat in [
            "Adaptive (Auto-Route)", "Baseline", "NLP-based",
            "Embedding-based", "LLM-based"
        ]:
            st.markdown(f"**{strat}**")
            res = run_strategy(strat, cmp_q, llm_ok)
            with st.spinner(f"{strat}: retrieve + rerank..."):
                if gen_answers:
                    pipe = run_answer_pipeline(
                        strat, cmp_q, res,
                        llm_available=answer_ok,
                        generate=True,
                    )
                    docs = pipe["reranked"]
                else:
                    cands = retrieve_candidates(strat, cmp_q, res)
                    docs = rerank_documents(cmp_q, cands, top_n=RERANK_TOP_N)
                    pipe = None
            label = (
                res.get("semantic_query", res.get("final_query", cmp_q))
                if strat == "LLM-based"
                else res.get("final_query", cmp_q)
            )
            st.text(str(label)[:200])
            if strat == "Adaptive (Auto-Route)":
                dec = res.get("decision", {})
                st.caption(
                    f"Action: `{res.get('action')}` · Reason: {dec.get('reason', '')}"
                )
            filters = res.get("filters") or {}
            if filters:
                extra = ", ".join(f"{k}={v}" for k, v in filters.items())
                prefer = res.get("prefer_source")
                if prefer:
                    extra += f", prefer={prefer}"
                st.caption(
                    f"Intent=`{res.get('intent', 'other')}` · Filters: {extra}"
                )
            if any(d["chunk_id"] in gold for d in docs):
                st.markdown("✓ GT found in top reranked")
            else:
                st.markdown("✗ GT missing from top reranked")
            if pipe is not None:
                st.write(pipe["generation"].get("answer", "")[:500])

with tab_results:
    router_results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "evaluation_results_router.json",
    )
    loaded_data = None
    if os.path.exists(router_results_path):
        with open(router_results_path, encoding="utf-8") as f:
            raw = json.load(f)
            loaded_data = raw.get("summary", raw)
    elif os.path.exists(EVAL_RESULTS):
        with open(EVAL_RESULTS, encoding="utf-8") as f:
            loaded_data = json.load(f)

    if loaded_data:
        rows = []
        for s, m in loaded_data.items():
            recall_data = m.get("recall", {})
            rows.append({
                "Strategy": s,
                "R@1": recall_data.get("@1", "-"),
                "R@3": recall_data.get("@3", "-"),
                "R@5": recall_data.get("@5", "-"),
                "MRR": m.get("mrr", "-"),
                "nDCG@5": m.get("ndcg@5", "-"),
                "Latency (ms)": m.get("mean_latency_ms", "-"),
            })
        st.subheader("Benchmark Comparison on Eval Dataset")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if "Adaptive Router (Proposed)" in loaded_data:
            dist = loaded_data["Adaptive Router (Proposed)"].get(
                "action_distribution_pct", {}
            )
            if dist:
                st.subheader("Adaptive Router Action Breakdown")
                d_cols = st.columns(len(dist))
                for col, (act, pct) in zip(d_cols, dist.items()):
                    col.metric(act, f"{pct}%")
    else:
        st.info("No results yet. Run `python scripts/evaluate_router.py --no-llm`.")

