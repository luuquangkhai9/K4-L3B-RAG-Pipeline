"""
Giao diện chat cho RAG pipeline.

Chủ đề: IELTS Writing — band descriptors, tiêu chí chấm điểm, bài viết mẫu.

Chạy: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="IELTS Writing RAG",
    page_icon="✍️",
    layout="wide",
)


# Nhãn hiển thị cho từng retrieval method theo contract.
METHOD_LABELS = {
    "hybrid": "🔀 Hybrid (dense + BM25 + RRF)",
    "dense": "🔵 Dense",
    "bm25": "🟠 BM25",
    "pageindex": "🟣 PageIndex (fallback)",
}

SOURCE_LABELS = {
    "hybrid": "Hybrid retrieval (dense + BM25, gộp bằng RRF)",
    "pageindex": "PageIndex fallback (vectorless)",
    "none": "Không tìm thấy nguồn phù hợp",
}


def render_sources(sources: list[dict]) -> None:
    """Hiển thị nguồn kèm method, score và URL để đối chiếu citation."""
    if not sources:
        return
    with st.expander(f"📚 Nguồn đã dùng ({len(sources)} chunks)", expanded=False):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            st.markdown(
                f"**[Document {index}]** {metadata.get('title', 'Không rõ tiêu đề')}  \n"
                f"`{METHOD_LABELS.get(source.get('retrieval_method'), source.get('retrieval_method'))}`"
                f" · score `{source.get('score', 0):.4f}`"
                f" · chunk `{metadata.get('chunk_index', '?')}`"
            )
            url = metadata.get("url")
            if url:
                st.caption(f"Nguồn: [{url}]({url})")
            else:
                st.caption(f"Nguồn: {metadata.get('source', 'n/a')}")
            with st.expander("Nội dung chunk"):
                st.text(source.get("content", ""))
            st.divider()


st.session_state.setdefault("messages", [])

with st.sidebar:
    st.title("✍️ IELTS Writing RAG")
    st.caption(
        "Hỏi đáp về **band descriptors, tiêu chí chấm điểm và bài viết mẫu** "
        "IELTS Writing, dựa trên tài liệu chính thức của ielts.org."
    )
    top_k = st.slider("Số chunks đưa vào context", 3, 10, 5)
    st.divider()
    st.markdown(
        "**Pipeline**\n\n"
        "1. ChromaDB dense search\n"
        "2. BM25 lexical search\n"
        "3. RRF fusion\n"
        "4. PageIndex fallback khi cosine thấp\n"
        "5. Sinh câu trả lời kèm citation"
    )
    if st.button("Xoá hội thoại"):
        st.session_state.messages = []
        st.rerun()

st.title("Hỏi đáp IELTS Writing")
st.caption("Câu trả lời chỉ dựa trên tài liệu đã thu thập; mỗi khẳng định đều kèm nguồn.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("retrieval_source"):
                st.caption(f"Truy xuất: {SOURCE_LABELS.get(message['retrieval_source'])}")
            render_sources(message.get("sources", []))

query = st.chat_input("Nhập câu hỏi, ví dụ: Band 7 cần đạt những tiêu chí nào ở Task Response?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và tổng hợp câu trả lời..."):
            from src.task10_generation import generate_with_citation

            try:
                result = generate_with_citation(query, top_k=top_k)
            except Exception as error:
                # Lỗi mạng/provider không được làm trắng màn hình chat.
                result = {
                    "answer": f"Không thể sinh câu trả lời: {type(error).__name__}.",
                    "sources": [],
                    "retrieval_source": "none",
                }

        st.markdown(result["answer"])
        st.caption(f"Truy xuất: {SOURCE_LABELS.get(result['retrieval_source'])}")
        render_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
    )