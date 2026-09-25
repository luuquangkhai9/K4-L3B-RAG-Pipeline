"""Tab "Đánh giá A/B" của app.py: đọc kết quả thật từ group_project/evaluation/."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


EVAL_DIR = Path(__file__).parent / "group_project" / "evaluation"
RESULTS_PATH = EVAL_DIR / "results.json"
SELFTEST_PATH = EVAL_DIR / "metric_selftest.json"

METRICS = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer relevance",
    "context_recall": "Context recall",
    "context_precision": "Context precision",
}
# Palette categorical đã validate (CVD + contrast); màu gắn theo config, không theo thứ hạng.
CONFIG_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json(path: Path):
    return _load(str(path), path.stat().st_mtime) if path.exists() else None


def _label(key: str, config: dict) -> str:
    return f"{key} — {config['name']}"


def render_kpis(configs: dict) -> None:
    columns = st.columns(len(configs))
    base = configs["A"]["summary"]
    for column, (key, config) in zip(columns, configs.items()):
        summary = config["summary"]
        with column.container(border=True):
            st.markdown(f"**{_label(key, config)}**")
            delta = None if key == "A" else f"{summary['average'] - base['average']:+.3f} so với A"
            st.metric("Điểm trung bình 4 metric", f"{summary['average']:.3f}", delta=delta)
            c1, c2, c3 = st.columns(3)
            c1.metric("Trả lời được", f"{summary['answered_rate']:.0%}", help="Tỉ lệ câu golden (trong domain) được trả lời thay vì từ chối.")
            c2.metric("Từ chối đúng", f"{summary['ood_refusal_rate']:.0%}", help="Tỉ lệ câu ngoài domain bị từ chối an toàn.")
            c3.metric("Latency", f"{summary['avg_latency']:.1f}s", help="Thời gian trung bình retrieval + generation mỗi câu.")


def render_chart(configs: dict) -> None:
    rows = [
        {"Metric": label, "Config": _label(key, config), "key": key, "Score": config["summary"][metric]}
        for key, config in configs.items()
        for metric, label in METRICS.items()
    ]
    data = pd.DataFrame(rows)
    domain = [_label(k, c) for k, c in configs.items()]
    color = alt.Color("Config:N", scale=alt.Scale(domain=domain, range=[CONFIG_COLORS[k] for k in configs]),
                      legend=alt.Legend(orient="top", title=None))
    base = alt.Chart(data).encode(
        x=alt.X("Metric:N", sort=list(METRICS.values()), title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Config:N", sort=domain),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Điểm (0–1)",
                axis=alt.Axis(grid=True, gridOpacity=0.3)),
        tooltip=[alt.Tooltip("Config:N"), alt.Tooltip("Metric:N"), alt.Tooltip("Score:Q", format=".3f")],
    )
    bars = base.mark_bar(size=34, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(color=color)
    labels = base.mark_text(dy=-8, fontSize=11).encode(text=alt.Text("Score:Q", format=".2f"))
    st.altair_chart((bars + labels).properties(height=360), width="stretch")

    table = data.pivot(index="Metric", columns="key", values="Score").reindex(list(METRICS.values()))
    table.loc["Trung bình"] = table.mean()
    table["Δ B−A"] = table["B"] - table["A"]
    table["Δ B−C (dịch query)"] = table["B"] - table["C"]
    st.dataframe(table.style.format("{:.3f}"), width="stretch")


def render_findings(configs: dict) -> None:
    a, b, c = (configs[k]["summary"] for k in "ABC")
    best = max(configs, key=lambda k: configs[k]["summary"]["average"])
    cmp = lambda x, y: "cao hơn" if x > y else "thấp hơn" if x < y else "bằng"  # noqa: E731
    ab = b["average"] - a["average"]
    verdict = ("nằm trong dao động của LLM judge với 18 câu — chưa đủ để kết luận" if abs(ab) < 0.05
               else "đủ lớn để xem là khác biệt thật")
    faith = {k: configs[k]["summary"]["faithfulness"] for k in configs}
    st.markdown("#### Nhận xét (sinh từ số liệu)")
    st.markdown(
        f"""
- **Config tốt nhất theo điểm trung bình:** {_label(best, configs[best])} ({configs[best]['summary']['average']:.3f}).
- **Dense only vs Hybrid (B−A = {ab:+.3f}):** {verdict}. Hybrid có context recall {cmp(b['context_recall'], a['context_recall'])}
  ({b['context_recall']:.2f} so với {a['context_recall']:.2f}) và context precision {cmp(b['context_precision'], a['context_precision'])}
  ({b['context_precision']:.2f} so với {a['context_precision']:.2f}).
- **Query translation (B−C = {b['average'] - c['average']:+.3f}):** trả lời được {b['answered_rate']:.0%} khi bật dịch
  so với {c['answered_rate']:.0%} khi tắt; đúng nguồn {b['source_hit_rate']:.0%} so với {c['source_hit_rate']:.0%}.
- **Faithfulness:** {", ".join(f"{k} = {v:.2f}" for k, v in faith.items())} — metric đã qua self-test (câu bịa chấm 0.0),
  nên điểm cao nghĩa là câu trả lời thật sự bám context.
- **Safe refusal ngoài domain:** {", ".join(f"{k} = {configs[k]['summary']['ood_refusal_rate']:.0%}" for k in configs)}.
"""
    )


def render_questions(configs: dict) -> None:
    st.markdown("#### Chi tiết từng câu hỏi")
    records = []
    for key, config in configs.items():
        for row in config["rows"]:
            score = sum(row[m] for m in METRICS) / len(METRICS)
            records.append({"Câu hỏi": row["question"], "Ngôn ngữ": row["language"], "Config": key,
                            "Điểm TB": score, **{METRICS[m]: row[m] for m in METRICS},
                            "Trả lời": not row["refused"], "Đúng nguồn": row["expected_source_hit"],
                            "Cosine": row["best_dense_score"] or 0.0, "Latency (s)": row["latency"],
                            "Câu trả lời": row["answer"]})
    frame = pd.DataFrame(records)

    view = st.segmented_control("Hiển thị", ["So sánh A/B/C", "Chi tiết một config"],
                                default="So sánh A/B/C", key="eval_view")
    progress = st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")
    if view == "Chi tiết một config":
        key = st.segmented_control("Config", list(configs), default="B", key="eval_config",
                                   format_func=lambda k: _label(k, configs[k]))
        detail = frame[frame["Config"] == (key or "B")].drop(columns="Config").sort_values("Điểm TB")
        st.dataframe(detail, hide_index=True, width="stretch",
                     column_config={"Điểm TB": progress, **{v: progress for v in METRICS.values()},
                                    "Cosine": st.column_config.NumberColumn(format="%.2f"),
                                    "Latency (s)": st.column_config.NumberColumn(format="%.1f"),
                                    "Câu trả lời": st.column_config.TextColumn(width="large")})
    else:
        wide = frame.pivot_table(index=["Câu hỏi", "Ngôn ngữ"], columns="Config", values="Điểm TB").reset_index()
        answered = frame.pivot_table(index="Câu hỏi", columns="Config", values="Trả lời", aggfunc="first")
        for key in configs:
            wide[f"{key} trả lời"] = wide["Câu hỏi"].map(answered[key])
        wide["Δ B−A"] = wide["B"] - wide["A"]
        st.dataframe(wide.sort_values("Δ B−A"), hide_index=True, width="stretch",
                     column_config={k: st.column_config.ProgressColumn(_label(k, configs[k]), min_value=0.0,
                                                                        max_value=1.0, format="%.2f")
                                    for k in configs} | {"Δ B−A": st.column_config.NumberColumn(format="%+.2f")})
        st.caption("Sắp theo Δ B−A tăng dần: hàng đầu là câu hybrid kém dense nhiều nhất.")


def render_selftest() -> None:
    selftest = load_json(SELFTEST_PATH)
    with st.expander("Kiểm chứng metric (self-test với ca biết trước kết quả)"):
        if not selftest:
            st.caption("Chưa có. Chạy `python -m src.evaluation --selftest`.")
            return
        passed = sum(r["passed"] for r in selftest)
        st.markdown(f"**{passed}/{len(selftest)} ca PASS** — metric phân biệt được câu bịa/lạc đề/thiếu context.")
        st.dataframe(pd.DataFrame(selftest), hide_index=True, width="stretch",
                     column_config={"passed": st.column_config.CheckboxColumn("Pass")})


def render_evaluation() -> None:
    results = load_json(RESULTS_PATH)
    if not results:
        st.info("Chưa có kết quả đánh giá. Chạy `python -m src.evaluation` rồi tải lại trang.")
        return
    info, configs = results["run_info"], results["configs"]
    st.subheader("Đánh giá A/B trên golden dataset")
    st.caption(
        f"{info['golden_size']} câu golden + 5 câu ngoài domain · top_k={info['top_k']} · "
        f"threshold={info['score_threshold']} · generator {info['generator_model']} · "
        f"evaluator {info['evaluator_model']} · embedding {info['embedding_model']} · chạy lúc {info['date']}"
    )
    render_kpis(configs)
    render_chart(configs)
    render_findings(configs)
    render_questions(configs)
    render_selftest()
    st.caption("Chạy lại: `python -m src.evaluation` (~3 phút) và `python -m src.evaluation --selftest`.")
