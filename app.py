# app.py
# ======
# Ce script fait 3 choses :
# 1. Charge un index FAISS qui contient nos documents (Wikipédia ici)
# 2. Branche un LLM gratuit (Llama-3-8B-Instruct via Groq) grâce à LangChain
# 3. Crée une interface Gradio où l'utilisateur tape une question et reçoit
#    une réponse générée + les sources utilisées

# ------------------------------------------------------------------
# 0. Imports standards
# ------------------------------------------------------------------
import os
from dotenv import load_dotenv  # charge les variables définies dans .env (clé API)

# ------------------------------------------------------------------
# 1. Imports LangChain : LLM + chaîne de réponse
# ------------------------------------------------------------------
from langchain_groq import ChatGroq  # wrapper Groq (LLM gratuit, rapide)
from langchain.chains import RetrievalQA  # chaîne "question → réponse" avec contexte
from langchain.prompts import PromptTemplate  # template pour dire au LLM comment répondre

# ------------------------------------------------------------------
# 2. Imports pour le vecteur-store (base de connaissances)
# ------------------------------------------------------------------
from langchain_community.vectorstores import FAISS  # index de similarité
from langchain_community.embeddings import HuggingFaceEmbeddings  # modèle d'embeddings

# ------------------------------------------------------------------
# 3. Imports UI
# ------------------------------------------------------------------
import gradio as gr

# ------------------------------------------------------------------
# 4. Chargement de la clé API Groq (dans .env, non commitée)
# ------------------------------------------------------------------
load_dotenv()  # lit le fichier .env local
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Clé GROQ_API_KEY manquante dans .env ou variables HF Spaces")

# ------------------------------------------------------------------
# 5. Initialisation du LLM
# ------------------------------------------------------------------
# ChatGroq : interface OpenAI-compatible → pas de carte bleue, 30 k tokens/h gratuits
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # modèle open-source hébergé par Groq
    temperature=0.3,  # 0 = très déterministe, 1 = très créatif
    groq_api_key=GROQ_API_KEY,
)

# ------------------------------------------------------------------
# 6. Template de prompt : on guide le LLM pour qu’il réponde en français
#    et qu’il reste concis en utilisant le contexte fourni
# ------------------------------------------------------------------
prompt_template = """Tu es un assistant scientifique francophone.
Réponds de manière concise et claire en utilisant uniquement le contexte ci-dessous.
Si le contexte ne permet pas de répondre, dis simplement « Je ne sais pas. »

Contexte :
{context}

Question : {question}
Réponse (3-4 phrases max) :"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# ------------------------------------------------------------------
# 7. Chargement de l’index FAISS (base de connaissances)
# ------------------------------------------------------------------
INDEX_DIR = "faiss_open_index"  # dossier créé précédemment
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

# allow_dangerous_deserialization=True car l’index a été sauvé localement
vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# ------------------------------------------------------------------
# 8. Chaîne de réponse : RetrievalQA
#    - retriever : trouve les 3 passages les plus proches
#    - llm : génère la réponse à partir de ces passages
# ------------------------------------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # on « bourre » tous les passages dans le prompt
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
)

# ------------------------------------------------------------------
# 9. Fonction appelée par Gradio
# ------------------------------------------------------------------
def answer_question(question: str) -> str:
    """Pose la question au LLM et renvoie la réponse + liens sources."""
    question = question.strip()
    if not question:
        return "<i>Entre une question…</i>"

    # 1) réponse générée
    answer = qa_chain.run(question)

    # 2) on récupère les documents utilisés pour afficher les sources
    docs = vectorstore.similarity_search(question, k=3)

    # 3) mise en forme HTML rapide
    sources = []
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title", "—")
        url = d.metadata.get("url", "#")
        snippet = (d.page_content[:300] + "…").replace("\n", " ")
        sources.append(
            f"<div style='margin:8px 0;padding:8px;border:1px solid #ddd;border-radius:8px'>"
            f"<b>{i}. {title}</b><br/>"
            f"<a href='{url}' target='_blank'>{url}</a><br/>"
            f"<span style='opacity:.8'>{snippet}</span>"
            f"</div>"
        )

    return f"<b>Réponse :</b><br/>{answer}<br/><br/><b>Sources :</b><br/>" + "\n".join(sources)

# ------------------------------------------------------------------
# 10. Interface Gradio
# ------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Experiment Brief Q&A") as demo:
    gr.Markdown("## 🔎 Experiment Brief — Q&R avec Llama-3 (Groq) + FAISS")
    with gr.Row():
        q = gr.Textbox(label="Ta question", placeholder="Ex. Quelle est la différence entre interleaving et A/B testing ?")
    go = gr.Button("Répondre")
    out = gr.HTML()
    go.click(answer_question, inputs=q, outputs=out)

# ------------------------------------------------------------------
# 11. Lancement
# ------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()