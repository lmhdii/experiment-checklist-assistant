# index_open_faiss.py — construit un index FAISS à partir du dataset open
from datasets import load_dataset, DatasetDict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
import re

DATASET_ID = "lmhdii/experiment-brief-open"  # ← laisse ton ID
INDEX_DIR = "faiss_open_index"

def chunk(text, size=900, overlap=150):
    text = re.sub(r"\s+", " ", text or "").strip()
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i += max(size - overlap, 1)
    return out

print("→ Loading dataset…")
dsd = DatasetDict()
for split in ["wiki_en", "wiki_fr"]:
    try:
        dsd[split] = load_dataset(DATASET_ID, split=split)
        print(f"  {split}: {len(dsd[split])} rows")
    except Exception as e:
        print(f"  skip {split} ({e})")

docs = []
for split, ds in dsd.items():
    for r in tqdm(ds, desc=f"chunk {split}"):
        meta = {
            "id": r["id"],
            "title": r["title"],
            "url": r["url"],
            "language": r["language"],
            "source_type": r["source_type"],
            "split": split,
        }
        for c in chunk(r["text"]):
            docs.append(Document(page_content=c, metadata=meta))

print(f"→ Total chunks: {len(docs)}")

# Multilingue FR/EN
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={"normalize_embeddings": True}
)

print("→ Building FAISS…")
vs = FAISS.from_documents(docs, emb)
vs.save_local(INDEX_DIR)
print(f"✅ Saved index to ./{INDEX_DIR}")

# Smoke test
q = "Qu'est-ce qu'un SRM en A/B testing et comment le diagnostiquer ?"
retriever = vs.as_retriever(search_kwargs={"k": 5})
hits = retriever.invoke(q)

print("\nTop-5 résultats :")
for i, d in enumerate(hits, 1):
    print(f"{i}. {d.metadata.get('title')} [{d.metadata.get('language')}] — {d.metadata.get('url')}")
    print("   ", d.page_content[:140].replace("\n", " "), "…")
