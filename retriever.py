import json, os, pickle
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")
INDEX_CACHE  = os.path.join(os.path.dirname(__file__), "faiss_index.pkl")


class CatalogRetriever:
    def __init__(self):
        self.catalog: list[dict] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.index: faiss.IndexFlatIP | None = None
        self._load()

    # Build / load

    def _doc_text(self, item: dict) -> str:
        parts = [
            item.get("name", ""),
            item.get("description", ""),
            item.get("test_type", ""),
            " ".join(item.get("competencies", [])),
            " ".join(item.get("job_levels", [])),
        ]
        return " ".join(p for p in parts if p)

    def _build_index(self):
        docs = [self._doc_text(item) for item in self.catalog]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        X = self.vectorizer.fit_transform(docs).toarray().astype("float32")
        X = normalize(X, norm="l2")
        dim = X.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(X)

    def _save_cache(self):
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "catalog": self.catalog}, f)

    def _load(self):
        with open(CATALOG_PATH) as f:
            self.catalog = json.load(f)

        if os.path.exists(INDEX_CACHE):
            try:
                with open(INDEX_CACHE, "rb") as f:
                    cache = pickle.load(f)
                self.vectorizer = cache["vectorizer"]
                docs = [self._doc_text(item) for item in self.catalog]
                X = self.vectorizer.transform(docs).toarray().astype("float32")
                X = normalize(X, norm="l2")
                dim = X.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(X)
                return
            except Exception as e:
                print(f"Cache load failed ({e}), rebuilding index")

        self._build_index()
        self._save_cache()

    # Query

    def search(self, query: str, k: int = 15) -> list[dict]:
        q_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        q_vec = normalize(q_vec, norm="l2")
        k = min(k, len(self.catalog))
        scores, indices = self.index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.catalog[idx])
            item["_score"] = float(score)
            results.append(item)
        return results

    def get_by_name(self, name: str) -> dict | None:
        """Exact or fuzzy name lookup for compare queries."""
        name_lower = name.lower()
        for item in self.catalog:
            if item["name"].lower() == name_lower:
                return item
        # Partial match fallback
        for item in self.catalog:
            if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
                return item
        return None

    def all_names(self) -> list[str]:
        return [item["name"] for item in self.catalog]

    def catalog_urls(self) -> set[str]:
        return {item["url"] for item in self.catalog}


_retriever: CatalogRetriever | None = None

def get_retriever() -> CatalogRetriever:
    global _retriever
    if _retriever is None:
        _retriever = CatalogRetriever()
    return _retriever


if __name__ == "__main__":
    r = get_retriever()
    print(f"Loaded {len(r.catalog)} assessments, index dim={r.index.d}")
    results = r.search("Java developer mid-level stakeholder communication", k=5)
    for res in results:
        print(f"[{res['_score']:.3f}] {res['name']} ({res['test_type']})")