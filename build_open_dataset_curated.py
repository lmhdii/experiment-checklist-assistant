# build_open_dataset_curated.py — Wikipédia FR/EN must-have (strict mais pragmatique)
from typing import List, Dict, Optional
import wikipediaapi
from datasets import Dataset, DatasetDict, Features, Value, Sequence

HF_USER = "lmhdii"
DS_NAME = f"{HF_USER}/experiment-brief-open"

# ----- Candidats par thème (on essaie dans l'ordre) -----
CANDIDATES_EN = {
    "A/B testing": ["A/B testing", "Split testing"],
    "Interleaving": ["Interleaving (information retrieval)", "Team-draft interleaving"],
    "Sequential analysis": ["Sequential analysis"],
    "False discovery rate": ["False discovery rate", "Benjamini–Hochberg procedure", "Benjamini-Hochberg procedure"],
    "Sample size": ["Sample size determination"],
    "Power": ["Power (statistics)"],
    "Non-inferiority": ["Non-inferiority trial"],
    "Equivalence": ["Equivalence test"],
    "Bandit": ["Multi-armed bandit"],
    "Thompson": ["Thompson sampling"],
    "Randomized": ["Randomized controlled trial", "Randomized experiment"],
    "Control": ["Scientific control", "Controlled experiment", "Control group"],
    "EN Interleaving" : ["Team-draft interleaving", "Interleaving (statistics)"],
}

CANDIDATES_FR = {
    "Test A/B": ["Test A/B"],
    "Analyse séquentielle": ["Analyse séquentielle"],
    "FDR": ["Taux de fausses découvertes", "Taux de fausse découverte"],
    "Benjamini-Hochberg": ["Procédure de Benjamini-Hochberg", "Procédure de Benjamini–Hochberg"],
    "Taille d'échantillon": ["Taille d'échantillon", "Échantillon (statistiques)"],
    "Puissance": ["Puissance statistique", "Puissance (statistique)"],
    "Non-infériorité": ["Essai de non-infériorité"],
    "Équivalence": ["Test d'équivalence (statistiques)", "Test d'équivalence"],
    "Bandit": ["Bandit manchot"],
    "Thompson": ["Échantillonnage de Thompson"],
    "Essai randomisé": ["Essai randomisé contrôlé"],
    "Témoin": ["Groupe témoin"],  # + proche de "Scientific control"
    "FR FDR" : ["Taux de fausses découvertes", "Taux de fausse découverte (statistiques)"],
    "FR Non-infériorité" : ["Essai de non-infériorité", "Essai de non-infériorité (statistiques)"],
    "FR Équivalence" : ["Test d'équivalence (statistiques)", "Test d'équivalence"],
}

FEATURES = Features({
    "id":          Value("string"),
    "source_type": Value("string"),
    "title":       Value("string"),
    "url":         Value("string"),
    "language":    Value("string"),
    "year":        Value("string"),
    "topics":      Sequence(Value("string")),
    "text":        Value("string"),
})

# --- Garde-fou "pertinence domaine" (un poil plus large) ---
KEYS_EN = [
    "a/b testing","split testing","online controlled experiment","interleaving",
    "information retrieval","sample ratio mismatch","srm","cuped","guardrail",
    "overall evaluation criterion","oec","sequential","false discovery rate",
    "benjamini","multi-armed bandit","thompson sampling","non-inferiority",
    "equivalence test","power (statistics)","sample size","scientific control",
    "controlled experiment","control group","randomized controlled trial"
]
KEYS_FR = [
    "test a/b","expérience contrôlée","essai randomisé","analyse séquentielle",
    "taux de fausses découvertes","benjamini","taille d'échantillon",
    "puissance (statistique)","puissance statistique","non-infériorité",
    "test d'équivalence","bandit manchot","échantillonnage de thompson",
    "groupe témoin","essai randomisé contrôlé"
]
BLOCK_TITLES = {
    "Audio Video Interleave","Desirable difficulty","Essais (Montaigne)",
    "Équivalent métabolique","Expérience de Stanford"
}

def relevant(title: str, text: str, lang: str) -> bool:
    if title in BLOCK_TITLES:
        return False
    t = (title or "").lower()
    x = (text or "").lower()
    keys = KEYS_EN if lang == "en" else KEYS_FR
    return any(k in t or k in x for k in keys)
def fetch_best(wiki, lang, candidates):
    # 1) essai strict + garde-fou
    for title in candidates:
        p = wiki.page(title)
        if p.exists() and (p.text or "").strip() and relevant(p.title, p.text, lang):
            return {
                "id": f"wiki::{lang}::{p.title}",
                "source_type": "wiki",
                "title": p.title,
                "url": p.fullurl,
                "language": lang,
                "year": "",
                "topics": [],
                "text": p.text or "",
            }
    # 2) fallback "force include" si la page existe mais le garde-fou est trop strict
    for title in candidates:
        p = wiki.page(title)
        if p.exists() and (p.text or "").strip():
            return {
                "id": f"wiki::{lang}::{p.title}",
                "source_type": "wiki",
                "title": p.title,
                "url": p.fullurl,
                "language": lang,
                "year": "",
                "topics": [],
                "text": p.text or "",
            }
    return None

def collect(lang: str, topics: Dict[str,List[str]]) -> List[Dict]:
    wiki = wikipediaapi.Wikipedia(language=lang, user_agent="experiment-brief-assistant/0.4")
    out, seen = [], set()
    for _, cand in topics.items():
        row = fetch_best(wiki, lang, cand)
        if row and row["title"] not in seen:
            out.append(row); seen.add(row["title"])
            print(f"✓ [{lang}] {row['title']}")
        else:
            print(f"⚠️  missing: [{lang}] {cand}")
    return out

if __name__ == "__main__":
    print("→ Fetch curated EN (patched)…")
    en_rows = collect("en", CANDIDATES_EN)
    print("→ Fetch curated FR (patched)…")
    fr_rows = collect("fr", CANDIDATES_FR)

    wiki_en = Dataset.from_list(en_rows, features=FEATURES)
    wiki_fr = Dataset.from_list(fr_rows, features=FEATURES)
    dsd = DatasetDict({"wiki_en": wiki_en, "wiki_fr": wiki_fr})
    print({k: len(v) for k, v in dsd.items()})

    print(f"→ Push to Hub: {DS_NAME}")
    dsd.push_to_hub(DS_NAME, private=False)
    print("✅ Dataset publié (curated patched).")
