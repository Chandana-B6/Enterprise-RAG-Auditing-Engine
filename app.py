
import streamlit as st
import re
import io
import html
import hashlib
import pandas as pd

from ingestion import DocumentProcessor
from vector_store import VectorIndex
from compliance_engine import ComplianceAuditor

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise RAG Auditing Engine",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Enterprise RAG Auditing & Compliance Engine")
st.caption(
    "Document-grounded policy retrieval, source evidence, "
    "and compliance risk monitoring."
)

# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------

if "index" not in st.session_state:
    st.session_state.index = VectorIndex()

if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()

if "auditor" not in st.session_state:
    st.session_state.auditor = ComplianceAuditor()

if "audit_logs" not in st.session_state:
    st.session_state.audit_logs = []

if "indexed_document_name" not in st.session_state:
    st.session_state.indexed_document_name = None

if "indexed_document_hash" not in st.session_state:
    st.session_state.indexed_document_hash = None


# --------------------------------------------------
# 3. SAMPLE PDF GENERATOR
# --------------------------------------------------

def generate_sample_pdf_bytes():
    """Generate a sample enterprise policy PDF in memory."""

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PolicyTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=24,
        spaceAfter=16,
        textColor=colors.HexColor("#1A365D"),
    )

    heading_style = ParagraphStyle(
        "PolicyHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#2B6CB0"),
    )

    body_style = ParagraphStyle(
        "PolicyBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    story = [
        Paragraph(
            "NEXUS ENTERPRISE OPERATIONAL POLICY",
            title_style,
        ),
        Paragraph(
            "1. Remote Work and Telecommuting",
            heading_style,
        ),
        Paragraph(
            "Employees may work remotely up to three days per week.",
            body_style,
        ),
        Paragraph(
            "Core collaborative operational hours are "
            "10:00 AM to 4:00 PM IST.",
            body_style,
        ),
        Paragraph(
            "Team members must maintain active communication "
            "through corporate messaging platforms.",
            body_style,
        ),
        Paragraph(
            "Remote work requests must be submitted to department "
            "heads at least 48 hours in advance.",
            body_style,
        ),
        Paragraph(
            "2. Annual Leave and Time Off",
            heading_style,
        ),
        Paragraph(
            "Standard full-time employees receive 25 days of paid "
            "annual leave per calendar year.",
            body_style,
        ),
        Paragraph(
            "Leave requests must be scheduled and approved through "
            "the HR operational dashboard.",
            body_style,
        ),
        Paragraph(
            "Vacation scheduling should account for critical "
            "deployment cycles and business continuity.",
            body_style,
        ),
        Paragraph(
            "3. Information Security",
            heading_style,
        ),
        Paragraph(
            "Employees must protect confidential company information "
            "and use approved systems to access corporate data.",
            body_style,
        ),
        Paragraph(
            "4. Policy Verification",
            heading_style,
        ),
        Paragraph(
            "Questions about requirements not explicitly addressed "
            "in this sample policy must be referred to the relevant "
            "department for clarification.",
            body_style,
        ),
    ]

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()


# --------------------------------------------------
# 4. FREE EXTRACTIVE RETRIEVAL RESPONSE
# --------------------------------------------------

def normalize_word(word):
    """Normalize a word for basic keyword comparison."""

    word = word.lower().strip(".,!?;:()[]{}\"'")

    # Light normalization only; avoid aggressive stemming.
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 4 and word.endswith("s"):
        word = word[:-1]

    return word


STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "to", "of",
    "in", "for", "on", "at", "by", "with", "what",
    "when", "where", "who", "how", "why", "can",
    "could", "do", "does", "are", "was", "were",
    "be", "been", "being", "i", "me", "my", "we",
    "our", "please", "tell", "explain", "according",
    "policy", "company", "document",
}


def extractive_answer(question, context, max_sentences=3):
    """
    Return relevant sentences from retrieved source context.

    This is extractive retrieval, not generative AI. The function
    does not invent a new answer, and it abstains when it cannot
    find sufficient keyword overlap.
    """

    if not context:
        return (
            "I cannot verify this claim against the retrieved "
            "company policy context."
        ), [], 0.0

    query_words = {
        normalize_word(word)
        for word in re.findall(r"\b[a-zA-Z0-9]+\b", question)
        if word.lower() not in STOPWORDS
    }

    if not query_words:
        return (
            "Please enter a more specific policy question."
        ), [], 0.0

    candidates = []

    for chunk_index, chunk in enumerate(context):
        if not isinstance(chunk, str):
            chunk = str(chunk)

        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            chunk,
        )

        for sentence in sentences:
            sentence = sentence.strip(" \t\r\n•-*")

            if len(sentence) < 12:
                continue

            sentence_words = {
                normalize_word(word)
                for word in re.findall(
                    r"\b[a-zA-Z0-9]+\b",
                    sentence,
                )
            }

            overlap = query_words.intersection(sentence_words)

            if not overlap:
                continue

            # Keyword coverage is a retrieval heuristic, not a
            # probability that the answer is factually correct.
            score = len(overlap) / len(query_words)

            candidates.append(
                {
                    "score": score,
                    "sentence": sentence,
                    "chunk_index": chunk_index,
                    "overlap": len(overlap),
                }
            )

    if not candidates:
        return (
            "I cannot verify this claim against the retrieved "
            "policy context. No sufficiently relevant sentence "
            "was found. Try rephrasing the question or indexing "
            "the appropriate document."
        ), [], 0.0

    candidates.sort(
        key=lambda item: (
            item["score"],
            item["overlap"],
        ),
        reverse=True,
    )

    selected = []
    seen = set()

    for candidate in candidates:
        sentence = candidate["sentence"]
        key = sentence.casefold()

        if key in seen:
            continue

        selected.append(candidate)
        seen.add(key)

        if len(selected) >= max_sentences:
            break

    answer_text = "\n\n".join(
        item["sentence"] for item in selected
    )

    # This score is keyword coverage only. It is not a confidence
    # score, hallucination probability, or factuality guarantee.
    coverage = max(
        item["score"] for item in selected
    )

    return answer_text, selected, coverage


# --------------------------------------------------
# 5. SIDEBAR: SAMPLE PDF AND DOCUMENT INGESTION
# --------------------------------------------------

with st.sidebar:
    st.header("📋 Sample Testing Asset")
    st.caption(
        "Download a sample policy PDF to test the application."
    )

    st.download_button(
        label="📥 Download Sample Policy PDF",
        data=generate_sample_pdf_bytes(),
        file_name="Nexus_Corporate_Policy.pdf",
        mime="application/pdf",
        key="download_sample_pdf",
    )

    st.divider()

    st.header("📄 Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload a corporate policy PDF",
        type=["pdf"],
        key="policy_uploader",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        st.caption(
            f"Selected file: {uploaded_file.name}"
        )

        if st.button(
            "Process and Index Document",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Extracting text and creating document embeddings..."
                ):
                    # Use a unique temporary filename for each file.
                    temp_path = (
                        f"/tmp/policy_{file_hash[:16]}.pdf"
                    )

                    with open(temp_path, "wb") as file:
                        file.write(file_bytes)

                    processed_data = (
                        st.session_state.processor.process_document(
                            temp_path
                        )
                    )

                    if not processed_data:
                        st.warning(
                            "No readable text was extracted. "
                            "Check whether the PDF contains selectable "
                            "text or requires OCR."
                        )
                    else:
                        # IMPORTANT:
                        # This assumes upsert_documents adds or updates
                        # documents in the existing vector index.
                        # If it appends old documents, the vector-store
                        # implementation should be updated to support
                        # document replacement or collection reset.
                        st.session_state.index.upsert_documents(
                            processed_data
                        )

                        st.session_state.indexed_document_name = (
                            uploaded_file.name
                        )
                        st.session_state.indexed_document_hash = (
                            file_hash
                        )

                        st.success(
                            f"Processed {len(processed_data)} "
                            "document chunks successfully."
                        )

                        st.info(
                            "Document text was indexed. Ask a question "
                            "in the retrieval panel to inspect matching "
                            "source evidence."
                        )

            except Exception as exc:
                st.error(
                    "Document processing failed. Please check the "
                    "PDF and application logs."
                )
                st.exception(exc)

    if st.session_state.indexed_document_name:
        st.success(
            "Indexed document: "
            + st.session_state.indexed_document_name
        )


# --------------------------------------------------
# 6. KNOWLEDGE RETRIEVAL
# --------------------------------------------------

st.header("🔎 Audited Knowledge Retrieval")

query = st.text_input(
    "Enter an employee or client policy question",
    placeholder="Example: How many days of annual leave are provided?",
)

if st.button("Run Retrieval and Audit", type="primary"):
    st.session_state["submitted_query"] = query.strip()

active_query = st.session_state.get("submitted_query", "")

if active_query:
    if not st.session_state.indexed_document_name:
        st.warning(
            "Please upload and process a policy PDF in the sidebar first."
        )

    else:
        try:
            with st.spinner(
                "Searching indexed context and evaluating the response..."
            ):
                query_vector = (
                    st.session_state.processor.embedding_model
                    .encode(active_query)
                    .tolist()
                )

                retrieved_chunks = (
                    st.session_state.index.query_context(
                        query_vector,
                        n_results=2,
                    )
                )

            if isinstance(retrieved_chunks, str):
                retrieved_chunks = [retrieved_chunks]

            if not retrieved_chunks:
                st.warning(
                    "No relevant source chunks were retrieved. "
                    "Try another question or reprocess the document."
                )

            else:
                ai_response, selected_sentences, coverage = (
                    extractive_answer(
                        active_query,
                        retrieved_chunks,
                    )
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📄 Retrieved Answer")

                    st.info(ai_response)

                    st.caption(
                        "Answer mode: extractive retrieval. "
                        "No external LLM was used."
                    )

                    with st.expander(
                        "🔍 View Retrieved Source Context",
                        expanded=True,
                    ):
                        for idx, chunk in enumerate(
                            retrieved_chunks,
                            start=1,
                        ):
                            st.markdown(
                                f"**Retrieved chunk {idx}**"
                            )
                            st.write(chunk)

                    with st.expander(
                        "📌 View Answer Evidence"
                    ):
                        if selected_sentences:
                            for item in selected_sentences:
                                st.markdown(
                                    f"- {item['sentence']}"
                                )
                                st.caption(
                                    "Source: retrieved chunk "
                                    f"{item['chunk_index'] + 1}"
                                )
                        else:
                            st.write(
                                "No supporting sentence was selected."
                            )

                with col2:
                    st.subheader("📊 Compliance Audit Dashboard")

                    # Use the existing auditor, but do not assume
                    # its score is a calibrated probability.
                    try:
                        audit = (
                            st.session_state.auditor.audit_response(
                                active_query,
                                retrieved_chunks,
                                ai_response,
                            )
                        )

                        hallucination_score = float(
                            audit.get("hallucination_score", 0.0)
                        )
                        hallucination_score = min(
                            1.0,
                            max(0.0, hallucination_score),
                        )

                        ethical_violation = bool(
                            audit.get("ethical_violation", False)
                        )

                        flagged_reasons = audit.get(
                            "flagged_reasons",
                            [],
                        )

                        if not isinstance(flagged_reasons, list):
                            flagged_reasons = [
                                str(flagged_reasons)
                            ]

                        st.metric(
                            "Auditor Risk Score",
                            f"{hallucination_score * 100:.1f}%",
                        )

                        st.caption(
                            "Score produced by your existing compliance "
                            "auditor. It is not a validated probability "
                            "of hallucination."
                        )

                        if ethical_violation:
                            st.error(
                                "Potential ethical policy violation flagged."
                            )
                        else:
                            st.success(
                                "No ethical violation was flagged by "
                                "the current auditor."
                            )

                        if flagged_reasons:
                            st.markdown("**Audit Findings**")

                            for reason in flagged_reasons:
                                st.markdown(
                                    "- "
                                    + html.escape(str(reason))
                                )
                        else:
                            st.write(
                                "No additional audit findings were returned."
                            )

                    except Exception as exc:
                        audit = {
                            "hallucination_score": None,
                            "ethical_violation": None,
                            "flagged_reasons": [
                                "Compliance audit could not be completed."
                            ],
                        }

                        st.error(
                            "The answer was retrieved, but the compliance "
                            "audit failed. Check compliance_engine.py."
                        )
                        st.exception(exc)

                # ------------------------------------------
                # 7. AUDIT LOGGING
                # ------------------------------------------

                st.divider()
                st.header("📊 Enterprise Audit Log Storage")

                score = audit.get("hallucination_score")

                if score is None:
                    risk_display = "Audit failed"
                else:
                    risk_display = (
                        f"{float(score) * 100:.1f}%"
                    )

                log_entry = {
                    "Timestamp": pd.Timestamp.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "Document": (
                        st.session_state.indexed_document_name
                        or "Unknown"
                    ),
                    "User_Query": active_query,
                    "Answer_Mode": "Extractive retrieval",
                    "Hallucination_Risk": risk_display,
                    "Ethical_Violation": (
                        "UNKNOWN"
                        if audit.get("ethical_violation") is None
                        else (
                            "YES"
                            if audit["ethical_violation"]
                            else "NO"
                        )
                    ),
                    "Flagged_Reasons": "; ".join(
                        str(reason)
                        for reason in audit.get(
                            "flagged_reasons",
                            [],
                        )
                    ) or "None",
                }

                # Log each distinct question once per current run.
                log_key = (
                    st.session_state.indexed_document_hash,
                    active_query,
                )

                if st.session_state.get("last_log_key") != log_key:
                    st.session_state.audit_logs.append(log_entry)
                    st.session_state.last_log_key = log_key

        except Exception:
            st.error(
                "Retrieval failed. Verify that the PDF was processed "
                "and that the embedding model and vector store are working."
            )
            st.exception(Exception)


# --------------------------------------------------
# 8. AUDIT LOG TABLE AND CSV EXPORT
# --------------------------------------------------

if st.session_state.audit_logs:
    st.divider()
    st.header("📚 Audit History")

    logs_df = pd.DataFrame(
        st.session_state.audit_logs
    )

    st.dataframe(
        logs_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="📥 Export Audit Log (CSV)",
        data=logs_df.to_csv(index=False).encode("utf-8"),
        file_name="compliance_audit_report.csv",
        mime="text/csv",
        key="export_audit_csv",
    )

else:
    st.caption(
        "No audit entries yet. Process a PDF and run a policy question "
        "to create an audit history entry."
    )
