from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "faiss_open_index"

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# allow_dangerous_deserialization est nécessaire pour recharger FAISS sauvegardé
vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": 5})

while True:
    try:
        q = input("\nTa question (ENTER pour quitter): ").strip()
        if not q:
            break
        hits = retriever.invoke(q)
        for i, d in enumerate(hits, 1):
            print(f"{i}. {d.metadata.get('title')} — {d.metadata.get('url')}")
            print("   ", d.page_content[:140].replace('\\n',' '), "…")
    except KeyboardInterrupt:
        break
