from fastembed import TextEmbedding
from qdrant_client.models import Filter, FieldCondition, DatetimeRange


class FedRetriever:
    def __init__(self, qdrant_client, collection_name: str, top_k: int = 3):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def retrieve(self, sub_queries: list[dict]) -> list[dict]:
        results = []
        for q in sub_queries:
            vector = next(iter(self.embedder.embed([q["search_query"]]))).tolist()
            response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=Filter(must=[
                    FieldCondition(key="date", range=DatetimeRange(gte=q["start_date"], lte=q["end_date"]))
                ]),
                limit=self.top_k * 20,
                with_payload=True,
            )
            hits = [
                hit for hit in response.points
                if len(hit.payload.get("content", "")) >= 100
            ][:self.top_k]
            documents = [
                {
                    "date": hit.payload.get("date"),
                    "type": hit.payload.get("type"),
                    "content": self._with_neighbors(hit),
                    "score": hit.score,
                }
                for hit in hits
            ]
            results.append({**q, "documents": documents})
        return results

    def _with_neighbors(self, hit) -> str:
        neighbors = self.qdrant_client.retrieve(
            collection_name=self.collection_name,
            ids=[hit.id - 1, hit.id + 1],
            with_payload=True,
        )
        same_doc = {
            p.id: p.payload["content"] for p in neighbors
            if p.payload.get("date") == hit.payload.get("date")
            and p.payload.get("type") == hit.payload.get("type")
        }
        same_doc[hit.id] = hit.payload["content"]
        return self._merge([same_doc[i] for i in sorted(same_doc)])

    @staticmethod
    def _merge(chunks: list[str]) -> str:
        merged = chunks[0]
        for chunk in chunks[1:]:
            overlap = next(
                (n for n in range(min(100, len(merged), len(chunk)), 9, -1)
                 if merged.endswith(chunk[:n])),
                0,
            )
            merged += chunk[overlap:] if overlap else "\n" + chunk
        return merged

    def build_context(self, results: list[dict]) -> str:
        sections = []
        for r in results:
            lines = [f"[查詢] {r['search_query']}（{r['start_date']} ~ {r['end_date']}）"]
            if not r["documents"]:
                lines.append("（此範圍內沒有找到文件）")
            for doc in r["documents"]:
                lines.append(f"[{doc['date']}｜{doc['type']}] {doc['content']}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)
