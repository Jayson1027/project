"""Step 3 檢索品質的離線評估：以文件為單位計算 Recall@K 與 nDCG@K。

用法（在專案根目錄執行）：
  1. 產生標註範本：python -m scripts.eval_retrieval template
     讀 data/eval/queries.json，把每個查詢時間範圍內的所有 Fed 文件列進
     data/eval/labels.json，並附上該文件與查詢最相近的段落供判斷。
  2. 人工標註：在 labels.json 每份文件的 "grade" 填 2 / 1 / 0
     2 = 有段落直接回答查詢；1 = 部分相關；0 = 不相關
     也可以先用 Gemini 自動評分：python -m scripts.eval_retrieval judge
     （只填空白的 grade；人工修正時請把 grade_source 改成 "human"）
  3. 計算指標：python -m scripts.eval_retrieval evaluate --k 3
"""
import argparse
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import DatetimeRange, FieldCondition, Filter

from core.fed_retriever import FedRetriever

COLLECTION_NAME = "project"
QUERIES_PATH = Path("data/eval/queries.json")
LABELS_PATH = Path("data/eval/labels.json")
PREVIEW_CHUNKS = 5
GRADE_FIELDS = ("grade", "grade_source", "gemini_grade", "reason")


class RelevanceJudgement(BaseModel):
    grade: int = Field(description="相關程度：2 = 有段落直接回答查詢；1 = 部分相關；0 = 不相關", ge=0, le=2)
    reason: str = Field(description="一句話說明評分理由 (繁體中文)")


def doc_key(date: str, doc_type: str) -> str:
    return f"{date}|{doc_type}"


def recall_at_k(ranked: list[str], grades: dict[str, int], k: int, relevant_grade: int) -> float | None:
    relevant = {d for d, g in grades.items() if g >= relevant_grade}
    if not relevant:
        return None
    return len(relevant & set(ranked[:k])) / len(relevant)


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float | None:
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return None
    dcg = sum((2 ** grades.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    return dcg / idcg


def ranked_documents(retriever: FedRetriever, query: dict) -> list[str]:
    """跑和正式流程相同的檢索，把命中的段落換成文件，依首次出現順序去重。"""
    result = retriever.retrieve([query])[0]
    ranked = []
    for doc in result["documents"]:
        key = doc_key(doc["date"], doc["type"])
        if key not in ranked:
            ranked.append(key)
    return ranked


def build_template(qdrant: QdrantClient, retriever: FedRetriever) -> None:
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    existing = {}
    if LABELS_PATH.exists():
        for q in json.loads(LABELS_PATH.read_text(encoding="utf-8")):
            existing[q["id"]] = {
                d["doc"]: {f: d[f] for f in GRADE_FIELDS if f in d} for d in q["documents"]
            }

    output = []
    for q in queries:
        vector = next(iter(retriever.embedder.embed([q["search_query"]]))).tolist()
        date_filter = Filter(must=[
            FieldCondition(key="date", range=DatetimeRange(gte=q["start_date"], lte=q["end_date"]))
        ])
        total = qdrant.count(COLLECTION_NAME, count_filter=date_filter).count
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=date_filter,
            limit=total,
            with_payload=True,
        )
        best_chunks: dict[str, list[str]] = {}
        for hit in response.points:
            content = hit.payload.get("content", "")
            if len(content) < 100:
                continue
            chunks = best_chunks.setdefault(doc_key(hit.payload["date"], hit.payload["type"]), [])
            if len(chunks) < PREVIEW_CHUNKS:
                chunks.append(content)

        old = existing.get(q["id"], {})
        documents = [
            {"doc": key, "grade": None, **old.get(key, {}), "closest_passages": best_chunks[key]}
            for key in sorted(best_chunks)
        ]
        output.append({**q, "documents": documents})
        print(f"[{q['id']}] {len(documents)} 份文件待標註（{total} 個段落）")

    LABELS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"標註範本已寫到 {LABELS_PATH}，請在每份文件的 grade 填 2 / 1 / 0")


def judge(api_key: str) -> None:
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
        api_version="v1beta",
        timeout=120_000,
        retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2.0, max_delay=30.0, http_status_codes=[429, 503]),
    ))
    labeled = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    for q in labeled:
        todo = [d for d in q["documents"] if d.get("grade") is None]
        print(f"[{q['id']}] 需評分 {len(todo)} / {len(q['documents'])} 份文件")
        for d in todo:
            passages = "\n\n".join(f"<passage>{p}</passage>" for p in d["closest_passages"])
            prompt = f"""
            你是 Fed 文件檢索系統的相關性評審。請判斷這份 Fed 文件對下列檢索查詢的相關程度。

            【檢索查詢】: {q['search_query']}
            【查詢時間範圍】: {q['start_date']} ~ {q['end_date']}
            【文件】: {d['doc']}（以下是該文件中與查詢最相近的段落）
            {passages}

            【評分標準】：
            2 = 至少有一段直接討論查詢的核心主題，可作為該主題的歷史依據
            1 = 只提到相關背景或部分主題，無法單獨作為依據
            0 = 與查詢主題無關，只有泛泛的共同用詞
            請只根據提供的段落判斷，不要用你對該時期的其他知識補充。
            """
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RelevanceJudgement,
                    temperature=0,
                ),
            )
            result = RelevanceJudgement.model_validate_json(response.text)
            d.update(grade=result.grade, gemini_grade=result.grade, grade_source="gemini", reason=result.reason)
            print(f"   {d['doc']}: {result.grade}｜{result.reason}")
            LABELS_PATH.write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n評分完成，已寫回 {LABELS_PATH}。抽查時若修改 grade，請把 grade_source 改成 \"human\"。")


def print_agreement(labeled: list[dict]) -> None:
    pairs = [
        (d["grade"], d["gemini_grade"])
        for q in labeled for d in q["documents"]
        if d.get("grade_source") == "human" and d.get("gemini_grade") is not None
    ]
    if not pairs:
        print("人工抽查：尚無（把抽查過的文件 grade_source 改成 \"human\" 後會計算與 Gemini 的一致率）")
        return
    agree = sum(h == g for h, g in pairs)
    print(f"人工抽查：{len(pairs)} 份，與 Gemini 評分一致 {agree} 份（{agree / len(pairs):.0%}）")


def evaluate(retriever: FedRetriever, k: int, relevant_grade: int) -> None:
    labeled = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    recalls, ndcgs = [], []
    for q in labeled:
        grades = {d["doc"]: d["grade"] for d in q["documents"]}
        missing = [doc for doc, g in grades.items() if g is None]
        if missing:
            print(f"[{q['id']}] 略過：還有 {len(missing)} 份文件未標註")
            continue

        ranked = ranked_documents(retriever, q)
        recall = recall_at_k(ranked, grades, k, relevant_grade)
        ndcg = ndcg_at_k(ranked, grades, k)
        if recall is not None:
            recalls.append(recall)
        if ndcg is not None:
            ndcgs.append(ndcg)

        def fmt(v):
            return "—" if v is None else f"{v:.3f}"

        print(f"\n[{q['id']}] {q['search_query']}（{q['start_date']} ~ {q['end_date']}）")
        print(f"   檢索結果：{[f'{d} (grade {grades.get(d, 0)})' for d in ranked]}")
        print(f"   Recall@{k} = {fmt(recall)}　nDCG@{k} = {fmt(ndcg)}")

    print("\n=== 平均 ===")
    if recalls:
        print(f"Recall@{k}：{sum(recalls) / len(recalls):.3f}（{len(recalls)} 個查詢，grade >= {relevant_grade} 視為相關）")
    if ndcgs:
        print(f"nDCG@{k}：{sum(ndcgs) / len(ndcgs):.3f}（{len(ndcgs)} 個查詢）")
    if not recalls and not ndcgs:
        print("沒有可計算的查詢，請先完成標註。")
    print_agreement(labeled)


def main():
    parser = argparse.ArgumentParser(description="Step 3 檢索品質離線評估")
    parser.add_argument("command", choices=["template", "judge", "evaluate"])
    parser.add_argument("--k", type=int, default=3, help="取前幾份文件計分，也是檢索的 top_k")
    parser.add_argument("--relevant-grade", type=int, default=2, help="grade 達到多少算 Recall 的相關文件")
    args = parser.parse_args()

    load_dotenv(".env")
    if args.command == "judge":
        judge(os.environ["GEMINI_API_KEY"])
        return
    qdrant =QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=60)
    retriever = FedRetriever(qdrant, COLLECTION_NAME, top_k=args.k)

    if args.command == "template":
        build_template(qdrant, retriever)
    else:
        evaluate(retriever, args.k, args.relevant_grade)


if __name__ == "__main__":
    main()
