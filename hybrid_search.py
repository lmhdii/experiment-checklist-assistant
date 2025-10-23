# hybrid_search.py — Hybrid retrieval (FAISS dense + BM25 lexical) + query rewrite + domain filter + SRM boost
# Compatible Python 3.9

from typing import List, Tuple, Dict
from collections import defaultdict
import re

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# ---------- Config ----------
INDEX_DIR = "faiss_open_index"  # dossier créé par index_open_faiss.py
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Expansion de requêtes (ajoute du contexte domaine)
EXPAND: Dict[str, List[str]] = {
    r"\binterleaving\b": [
        "team-draft interleaving", "search ranking evaluation", "information retrieval"
    ],
    r"\ba/?b\b": [
        "ab testing", "split testing", "online controlled experiment", "randomized experiment"
    ],
    r"\bsrm\b|sample ratio mismatch": [
        "randomization check", "allocation imbalance", "allocation ratio",
        "chi-squared test", "goodness of fit", "pearson chi-squared", "A/A test"
    ],
    r"\bcuped\b": ["variance reduction", "covariate adjustment"],
    r"\bguardrail(s)?\b": ["guardrail metric", "overall evaluation criterion", "oec"],
    r"\bsequential\b": ["sequential analysis", "alpha spending", "group sequential"],
    r"\bfdr\b|\bfalse discovery rate\b": ["benjamini", "benjamini–hochberg"],
    r"\bbandit\b": ["multi-armed bandit", "thompson sampling"],
    r"\bnon[- ]?inferiority\b": ["equivalence test"],
}

# Termes de domaine pour filtrer les résultats trop génériques
DOMAIN_TERMS = [
    "a/b testing","ab testing","split testing","online controlled experiment",
    "interleaving","team-draft interleaving","information retrieval",
    "sample ratio mismatch","srm","randomization check","allocation ratio",
    "chi-squared","pearson","goodness of fit","a/a test",
    "cuped","guardrail","overall evaluation criterion","oec",
    "sequential analysis","alpha spending","false discovery rate","benjamini",
    "multi-armed bandit","thompson sampling","non-inferiority","equivalence test",
    "power analysis","sample size determination","control group","scientific control",
]

# Légers boosts pour les requêtes SRM
BOOST_TERMS = [
    "sample ratio mismatch","randomization check","allocation ratio",
    "chi-squared","goodness of fit","a/a test"
]

# ---------- Utilitaires ----------
def rewrite(q: str) -> str:
    """Ajoute des termes de domaine en fonction de la requête."""
    qn = q.lower()
    extra: List[str] = []
    for pat, terms in EXPAND.items():
        if re.search(pat, qn):
            extra += terms
    return q if not extra else f"{q} " + " ".join(extra)

def rrf(dense_hits: List[Tuple[Document, float]],
        sparse_hits: List[Document],
        k: int = 60,
        topk: int = 5) -> List[Document]:
    """Reciprocal Rank Fusion (RRF) — fusionne l'ordre dense et lexical."""
    score = defaultdict(float)

    for rank, (doc, _) in enumerate(dense_hits, start=1):
        score[id(doc)] += 1.0 / (k + rank)

    for rank, doc in enumerate(sparse_hits, start=1):
        score[id(doc)] += 1.0 / (k + rank)

    uniq: Dict[int, Document] = {}
    for (doc, _) in dense_hits:
        uniq[id(doc)] = doc
    for doc in sparse_hits:
        uniq[id(doc)] = doc

    ranked = sorted(uniq.values(), key=lambda d: score[id(d)], reverse=True)
    return ranked[:topk]

def is_domain(doc: Document) -> bool:
    """Filtre simple : conserve les docs contenant des termes de notre domaine."""
    hay = (doc.metadata.get("title", "") + " " + doc.page_content).lower()
    return any(t in hay for t in DOMAIN_TERMS)

def boost_rank(docs: List[Document]) -> List[Document]:
    """Boost très simple pour SRM (compte des occurrences)."""
    def score(doc: Document) -> int:
        txt = (doc.page_content + " " + doc.metadata.get("title","")).lower()
        return sum(txt.count(t) for t in BOOST_TERMS)
    return sorted(docs, key=score, reverse=True)

# ---------- Chargement index + BM25 ----------
def load_retrievers():
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    # FAISS inclut les chunks quand on sauvegarde via save_local
    vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)

    # Construire BM25 sur les mêmes documents (chunks)
    docs = list(vs.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 12  # top lexical

    return vs, bm25

# ---------- API de recherche ----------
class HybridSearcher:
    def __init__(self):
        self.vs, self.bm25 = load_retrievers()

    def search(self, q: str, k_dense: int = 12, k_final: int = 5):
        q2 = rewrite(q)

        # denses + scores
        dense = self.vs.similarity_search_with_score(q2, k=k_dense)

        # lexical — utiliser .invoke() (évite le warning deprecation)
        sparse = self.bm25.invoke(q2)

        fused = rrf(dense, sparse, topk=max(k_final * 3, 12))
        filtered = [d for d in fused if is_domain(d)]
        candidates = filtered or fused
        results = boost_rank(candidates)[:k_final]
        return results, q2

# ---------- CLI ----------
if __name__ == "__main__":
    try:
        hs = HybridSearcher()
        print("Hybrid search prêt ✅ (FAISS + BM25).")
        while True:
            q = input("\nTa question (ENTER pour quitter): ").strip()
            if not q:
                break
            hits, q2 = hs.search(q)
            print(f"\nQuery réécrite: {q2}")
            if not hits:
                print("Aucun résultat.")
                continue
            for i, d in enumerate(hits, 1):
                title = d.metadata.get("title")
                url = d.metadata.get("url")
                lang = d.metadata.get("language")
                snippet = d.page_content[:160].replace("\n", " ")
                print(f"{i}. {title} [{lang}] — {url}\n   {snippet} …")
    except KeyboardInterrupt:
        pass
