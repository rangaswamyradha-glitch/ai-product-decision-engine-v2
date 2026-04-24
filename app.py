# app.py — APDE v2 Full Demo
import os
import json as json_mod
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from pathlib import Path

# ── Load API key securely ──────────────────────────────────────────────────
# Local development: reads from .env file
# Streamlit Cloud: reads from Streamlit secrets
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

if hasattr(st, 'secrets') and "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error(
        "⛔ ANTHROPIC_API_KEY not found.\n\n"
        "**Local:** Check your .env file.\n\n"
        "**Streamlit Cloud:** Check your app Secrets in Settings."
    )
    st.stop()

st.set_page_config(
    page_title="AI Product Decision Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Imports ────────────────────────────────────────────────────────────────
from services.ingestion.mock_data import (
    load_all_demo_signals,
    generate_reviews,
    generate_tickets,
    generate_sales_signals,
    generate_market_signals,
    generate_internal_signals,
)
from services.nlp.embedder import SignalEmbedder
from services.nlp.synthesiser import FeatureSynthesiser
from services.nlp.hallucination_guard import HallucinationGuard
from services.scoring.confidence import calculate_confidence
from services.scoring.engine import ScoringEngine
from services.scoring.roi_model import monte_carlo_roi
from services.roadmap.generator import RoadmapGenerator

# ── Session state init ─────────────────────────────────────────────────────
if "embedder" not in st.session_state:
    with st.spinner(
        "⚙️ Waking up the engine — please wait 20–30 seconds..."
    ):
        try:
            embedder = SignalEmbedder()
            # Only reload signals if store is empty
            if embedder.count() == 0:
                signals = load_all_demo_signals()
                embedder.ingest_signals(signals)
            st.session_state.embedder = embedder
            st.session_state.signal_count = embedder.count()
        except Exception as e:
            st.error(
                f"⚠️ Engine startup error: {e}\n\n"
                "Please refresh the page to try again."
            )
            st.stop()

embedder = st.session_state.embedder

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 APDE v2")
    st.markdown("## 🎯 Business Objectives")

    arr = st.number_input(
        "Company ARR ($)", value=20_000_000,
        step=500_000, format="%d"
    )
    okr1 = st.text_input("OKR 1", "Reduce churn to <3.5% by Q4 2026")
    okr2 = st.text_input("OKR 2", "Grow enterprise ARR by $5M in FY26")
    okr3 = st.text_input("OKR 3", "Improve NPS from 32 to 45")
    okrs = [o for o in [okr1, okr2, okr3] if o]

    st.divider()
    st.markdown("## 📊 Signal Corpus")
    st.metric("Total Signals Loaded", st.session_state.signal_count)
    st.caption("5 sources: Reviews · Tickets · Sales · Market · Internal")

    if st.button("🔄 Refresh Demo Data"):
        embedder.clear()
        new_sigs = load_all_demo_signals()
        n = embedder.ingest_signals(new_sigs)
        st.session_state.signal_count = embedder.count()
        st.success(f"Refreshed: {n} signals")
        st.rerun()

    st.divider()
    st.markdown("## 📥 Upload Real Data (Optional)")
    csv_file = st.file_uploader("Upload reviews/tickets CSV", type="csv")
    if csv_file:
        df_upload = pd.read_csv(csv_file)
        st.write(f"Loaded {len(df_upload)} rows")
        if st.button("Ingest CSV"):
            from services.ingestion.base import Signal
            import uuid
            custom_signals = []
            for _, row in df_upload.iterrows():
                content = str(row.get(
                    "content", row.get(
                        "body", row.get(
                            "review", row.get("text", "")
                        )
                    )
                ))
                if content and content != "nan":
                    custom_signals.append(Signal(
                        id=str(uuid.uuid4()),
                        source_type=str(
                            row.get("source_type", "review")
                        ),
                        content=content,
                        metadata={"uploaded": "True"},
                    ))
            n = embedder.ingest_signals(custom_signals)
            st.success(f"✓ Ingested {n} custom signals")

# ── Main tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Signal Intelligence",
    "⚡ Feature Synthesis",
    "⚖️ Priority Matrix",
    "🗺️ ROI Roadmap",
    "🛡️ Explainability & Audit",
    "📂 Raw Signal Data",
])

# ── TAB 1: Signal Intelligence ─────────────────────────────────────────────
with tab1:
    st.header("Signal Intelligence Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📝 Reviews", "200", "↑ last 90d")
    col2.metric("🎫 Tickets", "150", "↑ last 60d")
    col3.metric("💼 Sales Signals", "60", "ARR-weighted")
    col4.metric("📊 Market + Internal", "60", "40 market + 20 internal")

    st.subheader("Semantic Signal Search")
    query = st.text_input(
        "Search signals (semantic)",
        placeholder="e.g. SSO, bulk export, data loss, slow dashboard"
    )
    source_filter = st.multiselect(
        "Filter by source",
        ["review", "ticket", "sales", "market", "internal"],
        default=["review", "ticket", "sales", "market", "internal"]
    )

    if query:
        # Pass 1: semantic search
        docs = embedder.retrieve(
            query, top_k=30, source_filter=source_filter
        )

        # Pass 2: theme-based search to guarantee market + internal
        theme_docs = embedder.query_by_theme(query, top_k=15)

        # Merge both passes — full deduplication by id
        all_results = {}
        for d in docs:
            all_results[d["id"]] = d
        for d in theme_docs:
            all_results[d["id"]] = d

        all_docs_list = list(all_results.values())
        all_source_types = list(set(
            d["source_type"] for d in all_docs_list
        ))

        conf = calculate_confidence(docs, all_source_types)

        st.info(
            f"**Confidence:** {conf.tier} "
            f"({conf.composite:.0%}) | "
            f"Found **{len(all_docs_list)} signals** across "
            f"**{len(all_source_types)} source types** "
            f"({', '.join(sorted(all_source_types))})"
        )

        # Show results sorted by similarity
        display_docs = sorted(
            all_docs_list,
            key=lambda x: x["similarity"],
            reverse=True
        )

        for d in display_docs:
            color = {
                "review":   "🔵",
                "ticket":   "🟡",
                "sales":    "🟢",
                "market":   "🟣",
                "internal": "🟠"
            }.get(d["source_type"], "⚪")

            with st.expander(
                f"{color} [{d['source_type'].upper()}] "
                f"(relevance: {d['similarity']:.0%}) — "
                f"{d['content'][:60]}..."
            ):
                st.write(d["content"])
                st.caption(
                    f"Source: {d['metadata'].get('source', 'unknown')} | "
                    f"Theme: {d['metadata'].get('theme', 'N/A')} | "
                    f"Date: {d.get('created_at', 'N/A')}"
                )

# ── TAB 2: Feature Synthesis ───────────────────────────────────────────────
with tab2:
    st.header("AI Feature Synthesis & Hypothesis Generation")
    st.info(
        "The engine analyses themes across all signal sources using RAG, "
        "generates evidence-grounded hypotheses, and checks for hallucinations."
    )

    default_themes = (
        "data loss on project switch\n"
        "SSO authentication\n"
        "bulk export functionality"
    )
    themes_raw = st.text_area(
        "Enter themes to analyse (one per line)",
        value=default_themes,
        height=120
    )
    themes = [t.strip() for t in themes_raw.strip().split("\n") if t.strip()]

    if st.button("🚀 Synthesise Features", type="primary"):
        if not okrs:
            st.error("Please enter at least one OKR in the sidebar first.")
        else:
            synthesiser = FeatureSynthesiser(embedder)
            results = []
            prog = st.progress(0)
            for i, theme in enumerate(themes):
                with st.spinner(f"Analysing: {theme}..."):
                    r = synthesiser.generate_hypothesis(theme, okrs)
                    # Run hallucination guard
                    docs = embedder.retrieve(theme, top_k=15)
                    guard = HallucinationGuard(docs)
                    r["verification"] = guard.verify(
                        r.get("hypothesis", "")
                    )
                    results.append(r)
                prog.progress((i + 1) / len(themes))

            st.session_state.hypotheses = results
            st.success(f"✓ Generated {len(results)} feature hypotheses")

    if "hypotheses" in st.session_state:
        for r in st.session_state.hypotheses:
            tier = r.get("confidence_tier", "?")
            icon = {
                "HIGH": "✅", "MEDIUM": "🟡",
                "LOW": "🟠", "INSUFFICIENT": "❌"
            }.get(tier, "❓")
            ver = r.get("verification", {})
            ver_icon = "🛡️ VERIFIED" if ver.get("passed") else "⚠️ FLAGGED"

            with st.expander(
                f"{icon} {r.get('feature_name', '?')} — "
                f"Confidence: {tier} | {ver_icon}"
            ):
                c1, c2, c3 = st.columns(3)
                conf = r.get("confidence", {})
                c1.metric("Signal Docs", r.get("retrieved_doc_count", 0))
                c2.metric(
                    "Composite Confidence",
                    f"{conf.get('composite', 0):.0%}"
                )
                c3.metric(
                    "Source Types",
                    len(r.get("source_types_used", []))
                )

                st.markdown(f"**Hypothesis:** {r.get('hypothesis', 'N/A')}")
                st.markdown(f"**OKR Alignment:** {r.get('okr_alignment', 'N/A')}")
                st.markdown(
                    f"**Source types:** "
                    f"{', '.join(r.get('source_types_used', []))}"
                )

                if not ver.get("passed"):
                    if ver.get("flagged_phrases"):
                        st.warning(
                            f"⚠️ Flagged phrases: "
                            f"{', '.join(ver['flagged_phrases'])}"
                        )
                    if ver.get("unverified_numbers"):
                        st.warning(
                            f"⚠️ Unverified numbers: "
                            f"{', '.join(ver['unverified_numbers'])}"
                        )
                if r.get("unsupported_claims"):
                    st.info(
                        f"📋 Unsupported claims removed: "
                        f"{'; '.join(r['unsupported_claims'])}"
                    )
                if r.get("analyst_note"):
                    st.caption(f"🔍 {r['analyst_note']}")

# ── TAB 3: Priority Matrix ─────────────────────────────────────────────────
with tab3:
    st.header("Priority Decision Matrix")
    if "hypotheses" not in st.session_state:
        st.info("👈 Go to Feature Synthesis tab first and generate hypotheses.")
    else:
        if st.button("⚖️ Score All Features", type="primary"):
            engine = ScoringEngine(okrs)
            scored = []
            for h in st.session_state.hypotheses:
                if h.get("confidence_tier") == "INSUFFICIENT":
                    continue
                with st.spinner(f"Scoring: {h.get('feature_name')}..."):
                    s = engine.score(h)
                    reach_pct = s.get("reach", 5) / 100
                    roi = monte_carlo_roi(
                        arr, reach_pct, s.get("effort_weeks", 5)
                    )
                    s["roi_base"]         = roi.base
                    s["roi_conservative"] = roi.conservative
                    s["roi_optimistic"]   = roi.optimistic
                    s["roi_net"]          = roi.net
                    s["dev_cost"]         = roi.dev_cost
                    s["payback_months"]   = roi.payback_months
                    s["roi_derivation"]   = roi.derivation
                    scored.append(s)

            st.session_state.scored = scored
            st.success(f"✓ Scored {len(scored)} features")

        if "scored" in st.session_state:
            scored = st.session_state.scored
            df_scored = pd.DataFrame([{
                "Feature":      s["feature_name"],
                "Score":        s["composite_score"],
                "Effort (wks)": s.get("effort_weeks", 5),
                "ROI ($K)":     round(s.get("roi_base", 0) / 1000, 1),
                "Confidence":   s.get("confidence_tier", "?"),
            } for s in scored])

            st.dataframe(
                df_scored.sort_values("Score", ascending=False),
                use_container_width=True
            )

            # 2x2 Priority Matrix
            color_map = {
                "HIGH": "#22C55E", "MEDIUM": "#F59E0B",
                "LOW": "#F97316", "INSUFFICIENT": "#F43F5E"
            }
            fig_matrix = go.Figure()
            for tier, color in color_map.items():
                sub = df_scored[df_scored["Confidence"] == tier]
                if len(sub) == 0:
                    continue
                fig_matrix.add_trace(go.Scatter(
                    x=sub["Effort (wks)"],
                    y=sub["Score"],
                    mode="markers+text",
                    name=f"{tier} confidence",
                    text=sub["Feature"],
                    textposition="top center",
                    marker=dict(
                        size=sub["ROI ($K)"].clip(8, 50),
                        color=color,
                        opacity=0.9
                    ),
                ))
            fig_matrix.add_hline(y=70, line_dash="dash", line_color="#4B5563")
            fig_matrix.add_vline(x=6, line_dash="dash", line_color="#4B5563")
            fig_matrix.update_layout(
                title="Value vs. Effort Matrix (bubble size = ROI)",
                xaxis_title="Effort (weeks)",
                yaxis_title="Composite Score (0–100)",
                height=450,
                plot_bgcolor="#1E293B",
                paper_bgcolor="#0F172A",
                font_color="#E2E8F0"
            )
            st.plotly_chart(fig_matrix, use_container_width=True)

            # SHAP breakdown
            st.subheader("Score Contribution Breakdown (SHAP)")
            for s in sorted(
                scored,
                key=lambda x: x.get("composite_score", 0),
                reverse=True
            ):
                shap = s.get("shap", {})
                cols = st.columns(6)
                cols[0].markdown(f"**{s['feature_name'][:20]}**")
                for i, (dim, val) in enumerate(shap.items()):
                    if i < 5:
                        cols[i + 1].metric(dim, f"{val:.1f}")

# ── TAB 4: ROI Roadmap ─────────────────────────────────────────────────────
with tab4:
    st.header("ROI Roadmap & P&L Impact")
    if "scored" not in st.session_state:
        st.info("👈 Score features in the Priority Matrix tab first.")
    else:
        scored = st.session_state.scored

        if st.button("🗺️ Generate Roadmap Narrative", type="primary"):
            gen = RoadmapGenerator()
            with st.spinner("Generating executive roadmap..."):
                roadmap = gen.generate(scored, okrs, arr)
            st.session_state.roadmap = roadmap

        if "roadmap" in st.session_state:
            rm = st.session_state.roadmap

            # Timeline columns
            col1, col2, col3 = st.columns(3)
            for col, key, label, color in [
                (col1, "q_now",   "🚀 Now (Q2 2026)",   "#4C9EEB"),
                (col2, "q_next",  "⏭ Next (Q3 2026)",  "#F59E0B"),
                (col3, "q_later", "🔮 Later (Q4 2026)", "#A78BFA"),
            ]:
                col.markdown(
                    f'<div style="background:{color}20;'
                    f'border:1px solid {color}44;'
                    f'border-radius:8px;padding:16px;'
                    f'text-align:center">'
                    f'<div style="color:{color};font-weight:700;'
                    f'font-size:11px;text-transform:uppercase;'
                    f'margin-bottom:6px">{label}</div>'
                    f'<div style="color:#E2E8F0;font-weight:700;'
                    f'font-size:15px">{rm.get(key, "TBD")}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.divider()
            st.subheader("Executive Narrative")
            st.markdown(
                f'<div style="background:#1E293B;'
                f'border-left:4px solid #4C9EEB;'
                f'padding:16px;border-radius:4px;'
                f'color:#CBD5E1;font-style:italic;'
                f'line-height:1.8">'
                f'{rm.get("exec_narrative", "")}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.divider()
            st.subheader("P&L Impact Table")
            pl_rows = []
            for i, s in enumerate(sorted(
                scored,
                key=lambda x: x.get("composite_score", 0),
                reverse=True
            )):
                pl_rows.append({
                    "Feature":               s["feature_name"],
                    "Quarter":               ["Q2 2026", "Q3 2026", "Q4 2026"][min(i, 2)],
                    "Revenue (Base)":        f"${s.get('roi_base', 0):,.0f}",
                    "Revenue (Conservative)":f"${s.get('roi_conservative', 0):,.0f}",
                    "Revenue (Optimistic)":  f"${s.get('roi_optimistic', 0):,.0f}",
                    "Dev Cost":              f"${s.get('dev_cost', 0):,.0f}",
                    "Net (Base)":            f"${s.get('roi_net', 0):,.0f}",
                    "Payback":               f"{s.get('payback_months', 0):.1f} months",
                })
            st.dataframe(pd.DataFrame(pl_rows), use_container_width=True)

            # ROI Waterfall chart
            total_rev  = sum(s.get("roi_base", 0) for s in scored)
            total_cost = sum(s.get("dev_cost", 0) for s in scored)
            fig_waterfall = go.Figure(go.Waterfall(
                name="P&L",
                orientation="v",
                x=[s["feature_name"] for s in scored] + ["Total Net"],
                y=[s.get("roi_base", 0) for s in scored] + [
                    total_rev - total_cost
                ],
                connector={"line": {"color": "#4B5563"}},
                increasing={"marker": {"color": "#22C55E"}},
                decreasing={"marker": {"color": "#F43F5E"}},
            ))
            fig_waterfall.update_layout(
                title="Cumulative ROI Waterfall",
                plot_bgcolor="#1E293B",
                paper_bgcolor="#0F172A",
                font_color="#E2E8F0",
                height=350
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

            # Human gates warning
            if rm.get("human_gates_needed"):
                st.warning(
                    "👤 **Human Gates Required Before Publishing:**\n" +
                    "\n".join(f"- {g}" for g in rm["human_gates_needed"])
                )

            # CSV Export
            gen2 = RoadmapGenerator()
            csv_data = gen2.to_csv(scored)
            st.download_button(
                "⬇️ Export Roadmap CSV (for Jira/Notion)",
                csv_data,
                "apde_roadmap.csv",
                "text/csv"
            )

# ── TAB 5: Explainability & Audit ──────────────────────────────────────────
with tab5:
    st.header("Explainability & Audit Trail")

    if "scored" not in st.session_state:
        st.info("👈 Complete scoring first.")
    else:
        scored = st.session_state.scored
        selected = st.selectbox(
            "Select a feature to inspect",
            [s["feature_name"] for s in scored]
        )
        feat = next(
            (s for s in scored if s["feature_name"] == selected), None
        )

        if feat:
            st.subheader(f"📋 {feat['feature_name']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Composite Score",
                      f"{feat.get('composite_score', 0):.1f}/100")
            c2.metric("Confidence", feat.get("confidence_tier", "?"))
            c3.metric("Effort", f"{feat.get('effort_weeks', '?')} weeks")
            c4.metric("ROI (Base)", f"${feat.get('roi_base', 0):,.0f}")

            st.subheader("Evidence Chain (SHAP Breakdown)")
            shap = feat.get("shap", {})
            if shap:
                fig_shap = px.bar(
                    x=list(shap.values()),
                    y=list(shap.keys()),
                    orientation="h",
                    color=list(shap.values()),
                    color_continuous_scale="Teal",
                    title="Score Contribution by Dimension",
                    labels={
                        "x": "Points contributed",
                        "y": "Dimension"
                    },
                )
                fig_shap.update_layout(
                    plot_bgcolor="#1E293B",
                    paper_bgcolor="#0F172A",
                    font_color="#E2E8F0"
                )
                st.plotly_chart(fig_shap, use_container_width=True)

            st.subheader("Score Rationale (AI-generated, evidence-cited)")
            for dim, text in feat.get("rationale", {}).items():
                st.markdown(f"**{dim.title()}:** {text}")

            st.subheader("ROI Derivation")
            for k, v in feat.get("roi_derivation", {}).items():
                st.markdown(f"- **{k}:** {v}")

            st.subheader("Counterfactual: What would change this ranking?")
            cf_col1, cf_col2 = st.columns(2)
            with cf_col1:
                new_effort = st.slider(
                    "If effort changed to (weeks):",
                    1.0, 26.0,
                    float(feat.get("effort_weeks", 5))
                )
            with cf_col2:
                new_reach = st.slider(
                    "If reach changed to (% users):",
                    0.01, 0.50, 0.05, 0.01,
                    format="%.0f%%"
                )

            cf_effort_score = max(1, 10 - (new_effort / 3))
            cf_composite = (
                feat.get("reach", 5)           * 2.0 +
                feat.get("impact", 5)           * 2.5 +
                feat.get("confidence_score", 5) * 2.0 +
                cf_effort_score                 * 2.0 +
                feat.get("strategic_fit", 5)    * 1.5
            )
            delta = cf_composite - feat.get("composite_score", 0)
            cf_roi = monte_carlo_roi(arr, new_reach, new_effort)

            st.metric(
                "Counterfactual Score",
                f"{cf_composite:.1f}/100",
                delta=f"{delta:+.1f} pts vs current"
            )
            st.metric(
                "Counterfactual ROI (Base)",
                f"${cf_roi.base:,.0f}",
                delta=f"${cf_roi.base - feat.get('roi_base', 0):+,.0f}"
            )

            st.subheader("Audit Log Entry")
            audit = {
                "feature":          feat["feature_name"],
                "composite_score":  feat.get("composite_score"),
                "confidence_tier":  feat.get("confidence_tier"),
                "shap":             feat.get("shap"),
                "roi_base":         feat.get("roi_base"),
                "hallucination_check": (
                    st.session_state.get("hypotheses", [{}])[0]
                    .get("verification", {})
                ),
                "model_used": (
                    "claude-haiku-4-5 (scoring) / "
                    "claude-sonnet-4-6 (narrative)"
                ),
                "note": "Human PM approval required before roadmap publish"
            }
            st.code(json_mod.dumps(audit, indent=2), language="json")

# ── TAB 6: Raw Signal Data ─────────────────────────────────────────────────
with tab6:
    st.header("📂 Raw Demo Signal Data")
    st.caption("470 synthetic signals across 5 sources — generated by mock_data.py")

    source = st.selectbox(
        "Select signal source to browse",
        ["All Sources", "Reviews (200)", "Tickets (150)",
         "Sales (60)", "Market (40)", "Internal (20)"]
    )

    @st.cache_data
    def get_all_signals_df():
        all_rows = []

        for s in generate_reviews(200):
            all_rows.append({
                "source_type": s.source_type,
                "content":     s.content,
                "platform":    str(s.metadata.get("platform", "")),
                "rating":      str(s.metadata.get("rating", "")),
                "theme":       str(s.metadata.get("theme", "")),
                "priority":    "",
                "csat":        "",
                "arr_at_risk": "",
                "created_at":  s.created_at.strftime("%Y-%m-%d"),
            })

        for s in generate_tickets(150):
            all_rows.append({
                "source_type": s.source_type,
                "content":     s.content,
                "platform":    "",
                "rating":      "",
                "theme":       str(s.metadata.get("theme", "")),
                "priority":    str(s.metadata.get("priority", "")),
                "csat":        str(s.metadata.get("csat", "")),
                "arr_at_risk": "",
                "created_at":  s.created_at.strftime("%Y-%m-%d"),
            })

        for s in generate_sales_signals(60):
            arr_val = s.metadata.get("arr_at_risk", 0)
            try:
                arr_str = f"${int(float(str(arr_val))):,}" if arr_val else ""
            except Exception:
                arr_str = str(arr_val)
            all_rows.append({
                "source_type": s.source_type,
                "content":     s.content,
                "platform":    str(s.metadata.get("source", "")),
                "rating":      "",
                "theme":       str(s.metadata.get("theme", "")),
                "priority":    "",
                "csat":        "",
                "arr_at_risk": arr_str,
                "created_at":  s.created_at.strftime("%Y-%m-%d"),
            })

        for s in generate_market_signals(40):
            all_rows.append({
                "source_type": s.source_type,
                "content":     s.content,
                "platform":    str(s.metadata.get("source", "")),
                "rating":      "",
                "theme":       str(s.metadata.get("theme", "")),
                "priority":    "",
                "csat":        "",
                "arr_at_risk": "",
                "created_at":  s.created_at.strftime("%Y-%m-%d"),
            })

        for s in generate_internal_signals(20):
            all_rows.append({
                "source_type": s.source_type,
                "content":     s.content,
                "platform":    str(s.metadata.get("source", "")),
                "rating":      "",
                "theme":       str(s.metadata.get("theme", "")),
                "priority":    "",
                "csat":        "",
                "arr_at_risk": "",
                "created_at":  s.created_at.strftime("%Y-%m-%d"),
            })

        df = pd.DataFrame(all_rows)

        # Force ALL columns to string to prevent pyarrow conflicts
        for col in df.columns:
            df[col] = df[col].fillna("").astype(str)
            df[col] = df[col].replace("nan", "")
            df[col] = df[col].replace("None", "")

        return df

    df_all = get_all_signals_df()

    # Filter by source
    if source == "Reviews (200)":
        df_show = df_all[df_all["source_type"] == "review"]
    elif source == "Tickets (150)":
        df_show = df_all[df_all["source_type"] == "ticket"]
    elif source == "Sales (60)":
        df_show = df_all[df_all["source_type"] == "sales"]
    elif source == "Market (40)":
        df_show = df_all[df_all["source_type"] == "market"]
    elif source == "Internal (20)":
        df_show = df_all[df_all["source_type"] == "internal"]
    else:
        df_show = df_all

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Reviews",  len(df_all[df_all["source_type"] == "review"]))
    c2.metric("Tickets",  len(df_all[df_all["source_type"] == "ticket"]))
    c3.metric("Sales",    len(df_all[df_all["source_type"] == "sales"]))
    c4.metric("Market",   len(df_all[df_all["source_type"] == "market"]))
    c5.metric("Internal", len(df_all[df_all["source_type"] == "internal"]))

    st.divider()

    # Theme breakdown chart
    if source in ["All Sources", "Reviews (200)", "Tickets (150)", "Sales (60)"]:
        theme_col = df_show[df_show["theme"] != ""]["theme"]
        if len(theme_col) > 0:
            theme_counts = theme_col.value_counts()
            fig_theme = px.bar(
                x=theme_counts.values,
                y=theme_counts.index,
                orientation="h",
                title="Signal Volume by Theme",
                labels={"x": "Number of signals", "y": "Theme"},
                color=theme_counts.values,
                color_continuous_scale="Blues",
            )
            fig_theme.update_layout(
                plot_bgcolor="#1E293B",
                paper_bgcolor="#0F172A",
                font_color="#E2E8F0",
                showlegend=False,
                height=300,
            )
            st.plotly_chart(fig_theme, use_container_width=True)

    # Full data table
    st.subheader(f"Showing {len(df_show)} signals")
    st.dataframe(
        df_show,
        use_container_width=True,
        height=400,
        column_config={
            "source_type": st.column_config.TextColumn("Source",         width="small"),
            "content":     st.column_config.TextColumn("Signal Content", width="large"),
            "theme":       st.column_config.TextColumn("Theme",          width="medium"),
            "rating":      st.column_config.TextColumn("Rating",         width="small"),
            "csat":        st.column_config.TextColumn("CSAT",           width="small"),
            "arr_at_risk": st.column_config.TextColumn("ARR at Risk",    width="medium"),
            "priority":    st.column_config.TextColumn("Priority",       width="small"),
            "created_at":  st.column_config.TextColumn("Date",           width="small"),
        }
    )

    # Download button
    csv_export = df_show.to_csv(index=False)
    st.download_button(
        "⬇️ Download signals as CSV",
        csv_export,
        f"apde_signals_{source.lower().replace(' ', '_')}.csv",
        "text/csv"
    )
