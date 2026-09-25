import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="IELTS Writing RAG",
    page_icon="✍️",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("IELTS Writing RAG")
    st.caption("Hỏi đáp dựa trên tài liệu IELTS Writing đã thu thập")
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("IELTS Writing assistant")
st.caption("Câu trả lời chỉ dựa trên nguồn trong corpus và kèm citation để đối chiếu.")


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Nguồn tham khảo ({len(sources)})"):
        for source in sources:
            metadata = source.get("metadata") or {}
            st.markdown(
                f"**[{source['id']}] {metadata.get('title', 'Tài liệu')}**  \n"
                f"`{metadata.get('source', 'unknown')}` · "
                f"{source.get('retrieval_method', 'unknown')} · score `{source.get('score', 0):.3f}`"
            )
            if metadata.get("url"):
                st.markdown(f"[Mở nguồn]({metadata['url']})")
            st.caption(source.get("content", "")[:600])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []))

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm nguồn và tạo câu trả lời..."):
            from src.task10_generation import generate_with_citation

            result = generate_with_citation(query, top_k=top_k)
        st.markdown(result["answer"])
        show_sources(result["sources"])
        st.caption(f"Retrieval: {result['retrieval_source']}")
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
    )
