import re
import time

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

from src.task9_retrieval_pipeline import SCORE_THRESHOLD  # noqa: E402
from src.task10_generation import LLM_MODEL, LLM_PROVIDER, generate_with_trace  # noqa: E402


st.set_page_config(
    page_title="IELTS Writing RAG",
    page_icon="📝",
    layout="wide",
)

EXAMPLES = [
    "What does a band 7 response need for Task Achievement in Task 1?",
    "Task 2 cần viết tối thiểu bao nhiêu từ?",
    "How is Coherence and Cohesion assessed?",
    "Mẹo tăng điểm IELTS Writing là gì?",
    "Will IELTS still offer a paper-based test?",
    "Giá vàng hôm nay bao nhiêu?",
]
METHOD_LABEL = {"hybrid": "Hybrid (Dense + BM25 → RRF)", "pageindex": "PageIndex fallback", "none": "Từ chối an toàn"}
_CITATION = re.compile(r"\[(\d+)\]")


@st.cache_data(show_spinner=False)
def corpus_stats() -> dict:
    from src.task4_chunking_indexing import get_collection

    metadatas = get_collection().get(include=["metadatas"])["metadatas"]
    documents = {(m["doc_type"], m["source"]) for m in metadatas}
    return {
        "chunks": len(metadatas),
        "legal": sum(1 for kind, _ in documents if kind == "legal"),
        "news": sum(1 for kind, _ in documents if kind == "news"),
    }


def ask(query: str, top_k: int, use_reranking: bool) -> dict:
    started = time.perf_counter()
    result, trace = generate_with_trace(query, top_k=top_k, use_reranking=use_reranking)
    return {
        "role": "assistant",
        "content": result["answer"],
        "result": result,
        "trace": trace,
        "latency": time.perf_counter() - started,
    }


def render_answer(message: dict) -> None:
    result = message["result"]
    answer = _CITATION.sub(r"**[\1]**", result["answer"])
    st.markdown(answer)
    if result["retrieval_source"] == "none":
        reason = message["trace"].get("refusal_reason") or ""
        st.caption(f"🛡️ Không đủ bằng chứng trong tài liệu — {reason}")
        return
    cited = message["trace"].get("cited") or []
    labels = [f"[{n}] {result['sources'][n - 1]['metadata']['title'][:60]}" for n in cited]
    st.caption("📚 Nguồn đã trích dẫn: " + (" · ".join(labels) if labels else "không có citation"))


def render_details(message: dict | None) -> None:
    st.subheader("Chi tiết truy xuất")
    if message is None:
        st.info("Đặt câu hỏi để xem nguồn, retrieval method và score tại đây.")
        return

    result, trace = message["result"], message["trace"]
    source = result["retrieval_source"]
    col1, col2, col3 = st.columns(3)
    # Contract chỉ cho retrieval_source là hybrid/pageindex/none; chế độ Dense only lấy từ trace.
    label = "Dense only" if source == "hybrid" and trace.get("mode") == "dense" else METHOD_LABEL.get(source, source)
    col1.metric("Retrieval", label.split(" (")[0])
    best = trace.get("best_dense_score")
    col2.metric(
        "Cosine cao nhất",
        f"{best:.3f}" if best is not None else "—",
        delta=f"ngưỡng {trace.get('score_threshold', SCORE_THRESHOLD):.2f}",
        delta_color="normal" if not trace.get("low_confidence") else "inverse",
    )
    col3.metric("Thời gian", f"{message['latency']:.1f}s")

    if trace.get("translated_query"):
        st.caption(f"🌐 Tìm thêm bằng bản dịch tiếng Anh: _{trace['translated_query']}_")
    if trace.get("fallback_attempted"):
        if trace.get("used") == "pageindex":
            st.success("Dense score dưới ngưỡng → đã dùng PageIndex fallback.")
        else:
            st.warning(f"Dense score dưới ngưỡng, PageIndex không dùng được: {trace.get('fallback_error')}")
    if trace.get("mode") == "dense":
        st.caption("Chế độ Dense only (không BM25/RRF).")

    sources = result["sources"]
    if not sources:
        st.caption("Không có nguồn nào được dùng.")
        return
    cited = set(trace.get("cited") or [])
    st.markdown(f"**{len(sources)} nguồn** (sắp theo score, [n] khớp citation trong câu trả lời)")
    for number, item in enumerate(sources, 1):
        metadata = item["metadata"]
        with st.container(border=True):
            mark = "✅ " if number in cited else ""
            st.markdown(f"{mark}**[{number}] {metadata['title']}**")
            ranks = item.get("source_ranks") or {}
            rank_text = " · ".join(f"{method} #{rank}" for method, rank in ranks.items())
            badges = [
                f":blue-badge[{metadata['doc_type']}]",
                f":violet-badge[{item['retrieval_method']}]",
                f":gray-badge[score {item['score']:.4f}]",
            ]
            if rank_text:
                badges.append(f":orange-badge[{rank_text}]")
            st.markdown(" ".join(badges))
            location = f"`{metadata['source']}` · chunk {metadata.get('chunk_index', '—')}"
            if metadata.get("url"):
                location += f" · [mở nguồn]({metadata['url']})"
            st.caption(location)
            with st.expander("Xem nội dung chunk"):
                st.text(item["content"])


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    st.title("📝 IELTS Writing RAG")
    st.caption(
        "Hỏi đáp về IELTS Writing: band descriptors, tiêu chí chấm, bài mẫu có nhận xét "
        "của giám khảo và các bài hướng dẫn của IELTS/IDP. Mọi câu trả lời đều có citation."
    )
    st.divider()
    st.subheader("Cấu hình")
    top_k = st.slider("Số chunks (top_k)", 3, 10, 5)
    mode = st.radio(
        "Retrieval",
        ["Hybrid (Dense + BM25 → RRF)", "Dense only"],
        help="Hybrid là cấu hình chính; Dense only để so sánh A/B.",
    )
    st.caption(f"Ngưỡng fallback (cosine): **{SCORE_THRESHOLD:.2f}** · LLM: **{LLM_PROVIDER} / {LLM_MODEL}**")
    st.divider()
    st.subheader("Kho tài liệu")
    try:
        stats = corpus_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Legal", stats["legal"])
        c2.metric("News", stats["news"])
        c3.metric("Chunks", stats["chunks"])
    except Exception as error:
        st.error(f"Chưa đọc được ChromaDB: {error}. Chạy `python -m src.task4_chunking_indexing`.")
    st.divider()
    st.subheader("Câu hỏi mẫu")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
    st.divider()
    if st.button("🗑️ Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Chatbot IELTS Writing")
st.caption("Trả lời chỉ từ tài liệu đã thu thập. Không đủ bằng chứng thì chatbot sẽ từ chối thay vì đoán.")

query = st.chat_input("Nhập câu hỏi về IELTS Writing...") or st.session_state.pending
st.session_state.pending = None

chat_col, detail_col = st.columns([3, 2], gap="large")

with chat_col:
    if not st.session_state.messages and not query:
        st.info("Chọn một câu hỏi mẫu ở thanh bên hoặc nhập câu hỏi bên dưới.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer(message)
            else:
                st.markdown(message["content"])

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Đang truy xuất và tạo câu trả lời..."):
                try:
                    answer = ask(query, top_k, use_reranking=mode.startswith("Hybrid"))
                except Exception as error:  # UI không được crash
                    answer = {
                        "role": "assistant",
                        "content": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
                        "result": {"answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
                                   "sources": [], "retrieval_source": "none"},
                        "trace": {"refusal_reason": f"Lỗi hệ thống: {error}"},
                        "latency": 0.0,
                    }
        st.session_state.messages.append(answer)
        st.rerun()

with detail_col:
    latest = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
    render_details(latest)
