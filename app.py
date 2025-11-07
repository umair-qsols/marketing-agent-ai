# app.py
import streamlit as st
from datetime import datetime
from generator import (
    question_chain,
    generate_final_doc,
    vectorstore,
)

st.set_page_config(page_title="Marketing Agent AI", layout="wide")
st.markdown(
    """
    <style>
    .lftfield {font-family: 'Arial'; color: #C00000; font-weight: bold; text-align: center; font-size: 2.5rem;}
    .subtitle {font-size: 1.1rem; color: #555; text-align: center;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Template selector
# -------------------------------------------------
template_names = sorted({
    meta.get("template_name", "Unknown")
    for meta in vectorstore._collection.get(include=["metadatas"])["metadatas"]
})

if not template_names:
    st.error("No templates found! Put **.docx** files in `./templates/` and restart.")
    st.stop()

st.title("Marketing Agent AI")

selected = st.selectbox("Choose a template", options=template_names)

# Client & Date
# client_name = st.text_input("Company Name", value="Montessori Mantra Teachers’ Training College")

# -------------------------------------------------
# Start Q&A
# -------------------------------------------------
if st.button("Start Q&A", type="primary"):
    with st.spinner("Generating questions…"):
        raw = question_chain.invoke({"template_name": selected})
        questions = [
            line.strip()[line.index(".") + 1 :].strip()
            for line in raw.split("\n")
            if line.strip() and line[0].isdigit()
        ]
    st.session_state.questions = questions
    st.session_state.answers = {}
    st.session_state.step = 0
    st.rerun()

# -------------------------------------------------
# Q&A Flow
# -------------------------------------------------
if "questions" in st.session_state:
    questions = st.session_state.questions
    answers = st.session_state.get("answers", {})
    step = st.session_state.get("step", 0)

    # ---------- LEFT SIDEBAR (Q&A recap with EDIT buttons) ----------
    with st.sidebar:
        st.markdown("## Q&A Recap")
        for i, q in enumerate(questions, 1):
            ans = answers.get(f"q{i}", "")
            st.markdown(f"**Q{i}.** {q}")
            if ans:
                st.success(ans[:150] + ("…" if len(ans) > 150 else ""))
            else:
                st.info("Pending")

            # EDIT BUTTON
            if st.button(f"Edit Q{i}", key=f"edit_{i}"):
                st.session_state.step = i - 1
                st.rerun()

            st.markdown("---")

    # ---------- MAIN AREA ----------
    col1, col2 = st.columns([3, 1])
    with col1:
        if step < len(questions):
            q = questions[step]
            st.markdown(f"### Q{step + 1}: {q}")

            answer_key = f"q{step + 1}"
            user_input = st.text_area(
                "Your answer:",
                value=answers.get(answer_key, ""),
                height=150,
                key=f"input_{step}",
                label_visibility="collapsed",
            )

            if st.button("Save & Next", type="primary"):
                answers[answer_key] = user_input.strip()
                st.session_state.answers = answers
                st.session_state.step = step + 1
                st.rerun()

        else:
            st.success("All questions answered!")
            if st.button("Generate Final Document"):
                with st.spinner("Building document…"):
                    docx_bytes, doc_md = generate_final_doc(selected, answers)
                    doc_md = doc_md.replace("<DATE>", datetime.now().strftime("%B %Y"))

                # -----------------------------------------------------------------
                # 1. Show markdown preview
                # -----------------------------------------------------------------
                st.markdown("## Final Document (preview)")
                st.markdown(doc_md)

                # -----------------------------------------------------------------
                # 2. Offer .docx download
                # -----------------------------------------------------------------
                st.download_button(
                    label="Download .docx",
                    data=docx_bytes,
                    file_name=f"{selected} {datetime.now().strftime('%Y%m%d')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )