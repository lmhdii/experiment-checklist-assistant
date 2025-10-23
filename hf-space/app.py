# app.py — UI Gradio simple (FAISS-only) avec citations cliquables
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "faiss_open_index"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_vs():
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    # L’index doit être présent dans ./faiss_open_index
    return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)

vs = load_vs()

def search(query: str, k: int, lang_filter: str):
    q = (query or "").strip()
    if not q:
        return "<i>Entre une question…</i>"
    docs = vs.similarity_search(q, k=int(k))
    # petit filtre langue (optionnel)
    if lang_filter in ("FR", "EN"):
        keep = "fr" if lang_filter == "FR" else "en"
        docs = [d for d in docs if (d.metadata.get("language","") == keep)] or docs

    html = []
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title", "—")
        url   = d.metadata.get("url", "#")
        lang  = d.metadata.get("language", "—")
        snippet = (d.page_content[:420] + "…").replace("\n", " ")
        html.append(
            f"<div style='margin:10px 0;padding:10px;border:1px solid #eee;border-radius:12px'>"
            f"<div><b>{i}. {title}</b> <span style='opacity:.6'>[{lang}]</span></div>"
            f"<div style='margin:4px 0'><a href='{url}' target='_blank'>{url}</a></div>"
            f"<div style='opacity:.85'>{snippet}</div>"
            f"</div>"
        )
    return "\n".join(html)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🔎 Experiment Brief — Recherche sourcée (FAISS)")
    with gr.Row():
        q = gr.Textbox(label="Ta question", placeholder="Ex. Différence interleaving vs A/B ?")
    with gr.Row():
        k = gr.Slider(1, 10, value=5, step=1, label="Nombre de passages (k)")
        lang = gr.Radio(choices=["Tous", "FR", "EN"], value="Tous", label="Langue")
    go = gr.Button("Rechercher")
    out = gr.HTML()
    go.click(search, inputs=[q, k, lang], outputs=out)

if __name__ == "__main__":
    demo.launch()
