"""
Enterprise Document Question-Answering Assistant — Streamlit UI.

Talks to the FastAPI RAG backend. Demo logins are shown on the sign-in screen.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8765")

st.set_page_config(
    page_title="Northstar Document Assistant",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROLE_LABELS = {
    "admin": "Administrator",
    "compliance": "Compliance Officer",
    "auditor": "Internal Auditor",
    "employee": "Employee",
}

ROLE_BLURB = {
    "admin": "Full corpus, ingest, and evaluation.",
    "compliance": "Policies, risk, and restricted security runbooks.",
    "auditor": "Policies, audit procedures, and risk reports. No Restricted IRP.",
    "employee": "Internal handbook and public/internal policies only.",
}


def api(method: str, path: str, **kwargs) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.request(
        method,
        f"{BACKEND_URL}{path}",
        headers=headers,
        timeout=60.0,
        **kwargs,
    )


def require_ok(response: httpx.Response) -> dict:
    try:
        payload = response.json()
    except Exception:
        payload = {"detail": response.text or "Unexpected server error."}
    if response.status_code >= 400:
        detail = payload.get("detail", payload)
        raise RuntimeError(str(detail))
    return payload


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f4f1ea; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background: #0f2744; }
        [data-testid="stSidebar"] * { color: #e8eef5 !important; }
        [data-testid="stSidebar"] .stMarkdown p { color: #cbd5e1 !important; }
        .hero {
            background: linear-gradient(120deg, #0f2744 0%, #1e3a5f 60%, #0f2744 100%);
            color: #f8fafc; padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.1rem;
        }
        .hero h1 { font-size: 1.7rem; margin: 0 0 .35rem 0; font-weight: 650; }
        .hero p { margin: 0; color: #cbd5e1; max-width: 52rem; }
        .metric-strip { display: flex; gap: 12px; flex-wrap: wrap; margin: .8rem 0 1.1rem; }
        .chip {
            background: #fff; border: 1px solid #e2d9c8; border-radius: 12px;
            padding: .65rem .9rem; min-width: 8.5rem;
        }
        .chip span { display: block; font-size: .72rem; color: #64748b; letter-spacing: .04em; text-transform: uppercase; }
        .chip strong { font-size: 1.15rem; color: #0f2744; }
        .cite {
            background: #fff; border-left: 4px solid #b45309; padding: .75rem 1rem;
            margin-bottom: .65rem; border-radius: 0 10px 10px 0;
        }
        .cite .meta { color: #9a3412; font-size: .82rem; font-weight: 650; }
        .warn {
            background: #fff7ed; border: 1px solid #fdba74; color: #9a3412;
            padding: .9rem 1rem; border-radius: 12px; margin: .6rem 0 1rem;
        }
        .ok {
            background: #ecfdf5; border: 1px solid #6ee7b7; color: #065f46;
            padding: .9rem 1rem; border-radius: 12px; margin: .6rem 0 1rem;
        }
        .login-card {
            background: #fff; border: 1px solid #e2d9c8; border-radius: 16px;
            padding: 1.4rem 1.5rem; box-shadow: 0 10px 30px rgba(15,39,68,.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def login_view() -> None:
    inject_css()
    st.markdown(
        """
        <div class="hero">
          <h1>Northstar Document Assistant</h1>
          <p>Ask grounded questions of company policies, audit procedures, risk reports,
          and cybersecurity documentation. Answers include page-level citations and
          abstain when the corpus does not contain enough evidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns((1.15, 1), gap="large")
    with left:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.subheader("Sign in")
        st.caption("Role-based permissions decide which documents you can search.")
        try:
            demo = require_ok(api("GET", "/options")).get("demo_users", [])
        except Exception:
            demo = []
            st.error(
                "Cannot reach the API. Start it with `uvicorn backend.main:app "
                "--host 0.0.0.0 --port 8765` from the repo root."
            )
        with st.form("login"):
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password", value="admin123")
            submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")
        if submitted:
            try:
                payload = require_ok(api("POST", "/auth/login", json={"username": username, "password": password}))
                st.session_state.token = payload["access_token"]
                st.session_state.user = payload
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("#### Demo accounts")
        st.caption("Use these credentials to exercise RBAC. Employees cannot see Confidential or Restricted files.")
        for account in demo:
            st.markdown(
                f"**{account['full_name']}** · `{account['username']}` / `{account['password']}`  \n"
                f"{ROLE_LABELS.get(account['role'], account['role'])} — "
                f"{ROLE_BLURB.get(account['role'], '')}"
            )


def sidebar(user: dict) -> dict:
    with st.sidebar:
        st.markdown("### Northstar")
        st.caption("Enterprise document Q&A")
        st.caption("Project by Mohsen-Fekri")
        st.markdown("---")
        st.markdown(f"**{user['full_name']}**")
        st.caption(ROLE_LABELS.get(user["role"], user["role"]))
        st.caption(ROLE_BLURB.get(user["role"], ""))
        visible = ", ".join(user.get("classifications") or [])
        st.caption(f"Visible classifications: {visible}")
        st.markdown("---")
        chunk_size = st.selectbox("Chunk size", [256, 512, 1024], index=1)
        method = st.selectbox(
            "Retrieval method",
            ["hybrid", "bm25", "tfidf"],
            format_func=lambda m: {"hybrid": "Hybrid (BM25 + TF-IDF)", "bm25": "BM25", "tfidf": "TF-IDF"}.get(m, m),
        )
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    return {"chunk_size": chunk_size, "method": method}


def ask_tab(settings: dict) -> None:
    st.markdown("#### Ask a grounded question")
    st.caption("The assistant cites page numbers from documents your role may access. It will refuse when evidence is weak.")
    examples = [
        "How often must employees rotate their passwords?",
        "What is the SLA for remediating critical audit findings?",
        "What is the residual risk rating for ransomware?",
        "What is the P1 incident response time?",
        "What is the CEO salary for 2026?",
    ]
    pick = st.selectbox("Try an example", ["—"] + examples)
    question = st.text_area(
        "Question",
        value="" if pick == "—" else pick,
        height=90,
        placeholder="Ask about a policy control, audit SLA, risk rating, or incident severity…",
    )
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving passages and grounding an answer…"):
            try:
                result = require_ok(
                    api(
                        "POST",
                        "/ask",
                        json={
                            "question": question.strip(),
                            "chunk_size": settings["chunk_size"],
                            "retrieval_method": settings["method"],
                        },
                    )
                )
            except Exception as exc:
                st.error(str(exc))
                return
        _render_answer(result)


def _render_answer(result: dict) -> None:
    if result["abstained"]:
        st.markdown(f'<div class="warn"><strong>Abstained</strong> — {result["answer"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ok"><strong>Answer</strong><br/>{result["answer"]}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", f"{result['confidence']:.0%}")
    c2.metric("Retrieval", result["retrieval_method"])
    c3.metric("Chunk size", result["chunk_size"])
    c4.metric("Generator", result["generator"])

    st.markdown("##### Page-level citations")
    if not result.get("citations"):
        st.caption("No citation passed the retrieval threshold.")
    for cite in result.get("citations", []):
        st.markdown(
            f'<div class="cite"><div class="meta">{cite["title"]} · {cite["filename"]} · page {cite["page"]} · score {cite["score"]:.2f}</div>'
            f"<div>{cite['snippet']}</div></div>",
            unsafe_allow_html=True,
        )

    with st.expander("Retrieved passages"):
        for passage in result.get("passages", []):
            st.markdown(
                f"**[{passage['rank']}] {passage['title']} · p. {passage['page']}** "
                f"(score {passage['score']:.2f}, {passage['classification']})"
            )
            st.write(passage["text"])


def library_tab(settings: dict, user: dict) -> None:
    try:
        payload = require_ok(api("GET", "/documents", params={"chunk_size": settings["chunk_size"]}))
    except Exception as exc:
        st.error(str(exc))
        return
    stats = payload.get("stats", {})
    st.markdown(
        f'<div class="metric-strip">'
        f'<div class="chip"><span>Documents</span><strong>{stats.get("documents", 0)}</strong></div>'
        f'<div class="chip"><span>Chunks</span><strong>{stats.get("chunks", 0)}</strong></div>'
        f'<div class="chip"><span>Pages</span><strong>{stats.get("pages", 0)}</strong></div>'
        f'<div class="chip"><span>Chunk size</span><strong>{stats.get("chunk_size", "")}</strong></div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    docs = payload.get("documents", [])
    if not docs:
        st.info("No documents are visible to your role yet.")
    else:
        st.dataframe(
            [
                {
                    "Title": d["title"],
                    "File": d["filename"],
                    "Domain": d["domain"].replace("_", " "),
                    "Classification": d["classification"],
                    "Pages": d["page_count"],
                }
                for d in docs
            ],
            use_container_width=True,
            hide_index=True,
        )

    if user["role"] in {"admin", "compliance"}:
        st.markdown("##### Ingest a PDF or Word file")
        uploaded = st.file_uploader("Upload", type=["pdf", "docx"])
        col_a, col_b, col_c = st.columns(3)
        title = col_a.text_input("Title")
        classification = col_b.selectbox("Classification", ["public", "internal", "confidential", "restricted"], index=1)
        domain = col_c.selectbox("Domain", ["company_policy", "audit_procedure", "risk_report", "cybersecurity"])
        if uploaded and st.button("Ingest document", type="primary"):
            try:
                require_ok(
                    api(
                        "POST",
                        "/documents/upload",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                        data={
                            "title": title,
                            "classification": classification,
                            "domain": domain,
                            "chunk_size": str(settings["chunk_size"]),
                        },
                    )
                )
                st.success(f"Ingested {uploaded.name}. It is now searchable for permitted roles.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    else:
        st.caption("Document ingest is limited to administrators and compliance officers.")


def evaluate_tab(settings: dict) -> None:
    st.markdown("#### Retrieval and grounding evaluation")
    st.caption(
        "Gold questions cover answerable policy items and should-abstain probes. "
        "Local metrics always run. RAGAS LLM-as-judge runs only when OPENAI_API_KEY is set."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Evaluate current settings", type="primary", use_container_width=True):
            with st.spinner("Scoring the gold set…"):
                try:
                    payload = require_ok(
                        api(
                            "POST",
                            "/evaluate",
                            params={
                                "chunk_size": settings["chunk_size"],
                                "retrieval_method": settings["method"],
                                "include_ragas": True,
                            },
                        )
                    )
                    st.session_state.eval_single = payload
                except Exception as exc:
                    st.error(str(exc))
    with col2:
        if st.button("Compare chunk sizes × retrievers", use_container_width=True):
            with st.spinner("Comparing 256 / 512 / 1024 with BM25, TF-IDF, and hybrid…"):
                try:
                    st.session_state.eval_compare = require_ok(api("POST", "/evaluate/compare"))
                except Exception as exc:
                    st.error(str(exc))

    if "eval_single" in st.session_state:
        local = st.session_state.eval_single.get("local", {})
        summary = local.get("summary", {})
        st.markdown("##### Current configuration")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Answer-context overlap", f"{summary.get('answer_context_overlap', 0):.2f}")
        m2.metric("Answer-reference overlap", f"{summary.get('answer_reference_overlap', 0):.2f}")
        m3.metric("Context recall", f"{summary.get('context_recall', 0):.2f}")
        m4.metric("Abstention acc.", f"{summary.get('abstention_accuracy', 0):.2f}")
        m5.metric("Local grounding", f"{summary.get('local_grounding_score', 0):.2f}")
        ragas = st.session_state.eval_single.get("ragas")
        if ragas:
            if ragas.get("available"):
                st.success(f"RAGAS scores: {ragas.get('ragas')}")
            else:
                st.info(ragas.get("reason", "RAGAS skipped."))
        st.dataframe(local.get("rows", []), use_container_width=True, hide_index=True)

    if "eval_compare" in st.session_state:
        compare = st.session_state.eval_compare
        best = compare.get("best") or {}
        st.markdown("##### Configuration sweep")
        if best:
            st.success(
                f"Best grounding: chunk size {best.get('chunk_size')} with "
                f"{best.get('retrieval_method')} (score {best.get('local_grounding_score')})."
            )
        st.dataframe(compare.get("runs", []), use_container_width=True, hide_index=True)


def main() -> None:
    inject_css()
    user = st.session_state.get("user")
    if not user:
        login_view()
        return
    settings = sidebar(user)
    st.markdown(
        f"""
        <div class="hero">
          <h1>Ask the corpus you are allowed to see</h1>
          <p>Signed in as {user['full_name']} ({ROLE_LABELS.get(user['role'], user['role'])}).
          Retrieval is filtered by classification. Answers are extractive unless an OpenAI key is configured.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tabs = ["Ask", "Library"]
    if user["role"] in {"admin", "auditor", "compliance"}:
        tabs.append("Evaluate")
    views = st.tabs(tabs)
    with views[0]:
        ask_tab(settings)
    with views[1]:
        library_tab(settings, user)
    if len(views) > 2:
        with views[2]:
            evaluate_tab(settings)


if __name__ == "__main__":
    main()
