import re
import time

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

from src.task9_retrieval_pipeline import SCORE_THRESHOLD  # noqa: E402
from src.task10_generation import LLM_MODEL, LLM_PROVIDER, generate_with_trace  # noqa: E402
from ui_evaluation import render_evaluation  # noqa: E402


st.set_page_config(
    page_title="IELTS Writing RAG",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Mỗi câu gợi ý kiểm tra một khả năng cụ thể của pipeline (lý do hiện ở tooltip).
EXAMPLE_GROUPS = {
    ":material/grading: Band descriptors & tiêu chí chấm": {
        "What does a band 7 response need for Task Achievement in Task 1?":
            "Tra bảng: kiểm tra chunking theo ô bảng (Task 1 × Band 7 × Task Achievement).",
        "Band 6 Lexical Resource trong Task 2 yêu cầu gì?":
            "Hỏi tiếng Việt về bảng tiếng Anh: kiểm tra query translation + tra bảng Task 2.",
        "How is Coherence and Cohesion assessed?":
            "Câu định nghĩa có ở nhiều tài liệu: kiểm tra RRF gộp nguồn và citation nhiều nguồn.",
        "Task 2 cần viết tối thiểu bao nhiêu từ?":
            "Số liệu chính xác (250 từ) chỉ có trong tài liệu tiếng Anh: từng bị từ chối trước khi thêm dịch query.",
    },
    ":material/description: Đề mẫu & nhận xét giám khảo": {
        "Why did the sample Task 2 response get band 5.5?":
            "Cần đúng đoạn nhận xét giám khảo: kiểm tra heading 'Examiner comment — Band' giữ nhãn band.",
        "Trong đề thi mẫu IELTS Academic Writing (large print), Task 1 yêu cầu mô tả bảng số liệu gì?":
            "Nội dung nằm giữa tài liệu đề thi: kiểm tra contextual chunk header (tên tài liệu trong mỗi chunk).",
        "What is the Task 2 essay topic in the IELTS sample question paper?":
            "Trích nguyên văn đề bài: kiểm tra câu trả lời trích đúng, không diễn giải sai.",
    },
    ":material/newspaper: Tin tức & mẹo luyện thi": {
        "Will IELTS still offer a paper-based test?":
            "Thông tin thời sự (bỏ thi giấy từ giữa 2026): kiểm tra nguồn news và link nguồn gốc.",
        "Mẹo tăng điểm IELTS Writing là gì?":
            "Câu mở, cần tổng hợp nhiều bài: kiểm tra câu trả lời có nhiều citation [n].",
        "Làm sao viết IELTS Writing rõ ràng và súc tích?":
            "Hỏi tiếng Việt, bài tiếng Việt: kiểm tra BM25 khớp từ khóa có dấu.",
    },
    ":material/block: Ngoài phạm vi (test từ chối an toàn)": {
        "Giá vàng hôm nay bao nhiêu?":
            "Hoàn toàn ngoài domain (cosine ~0.26 < ngưỡng): bị chặn ngay ở bước threshold, không gọi LLM.",
        "What is the TOEFL iBT speaking section format?":
            "Gần domain (kỳ thi tiếng Anh, cosine ~0.48 > ngưỡng): phải được LLM từ chối vì không có evidence.",
    },
}
TAB_CHAT = ":material/chat: Chatbot"
TAB_EVAL = ":material/analytics: Đánh giá A/B"
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


def queue_question(question: str) -> None:
    st.session_state.pending = question
    st.session_state.main_tab = TAB_CHAT  # bấm câu mẫu khi đang ở tab Đánh giá thì quay về chat


def render_suggestions() -> None:
    st.markdown("#### Câu hỏi gợi ý để test")
    st.caption("Bấm một câu để hỏi ngay, hoặc tự nhập câu hỏi ở ô chat bên dưới.")
    columns = st.columns(2)
    for index, (group, questions) in enumerate(EXAMPLE_GROUPS.items()):
        with columns[index % 2].container(border=True):
            st.markdown(f"**{group}**")
            for question, reason in questions.items():
                st.button(
                    question, key=f"suggest-{question}", width="stretch", help=reason,
                    on_click=queue_question, args=(question,),
                )


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
    for group, questions in EXAMPLE_GROUPS.items():
        with st.expander(group):
            for question, reason in questions.items():
                st.button(
                    question, key=f"side-{question}", width="stretch", help=reason,
                    on_click=queue_question, args=(question,),
                )
    st.divider()
    if st.button("Xoá hội thoại", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

st.title("Chatbot IELTS Writing")
st.caption("Trả lời chỉ từ tài liệu đã thu thập. Không đủ bằng chứng thì chatbot sẽ từ chối thay vì đoán.")

chat_tab, eval_tab = st.tabs([TAB_CHAT, TAB_EVAL], key="main_tab", on_change="rerun")

with eval_tab:
    if eval_tab.open:
        render_evaluation()

query = None
if chat_tab.open:
    query = st.chat_input("Nhập câu hỏi về IELTS Writing...") or st.session_state.pending
    st.session_state.pending = None

with chat_tab:
    chat_col, detail_col = st.columns([3, 2], gap="large")

with chat_col:
    if not st.session_state.messages and not query:
        render_suggestions()
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
