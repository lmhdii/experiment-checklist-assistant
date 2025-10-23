[![HF Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/lmhdii/experiment-checklist-assistant)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

# 🧪 Experiment Checklist Assistant (v0.1.0)

**Demo live →** [Hugging Face Space 🚀](https://huggingface.co/spaces/lmhdii/experiment-checklist-assistant)

Assistant bilingue *(FR/EN)* pour la **recherche et l’expérimentation en ligne** (A/B testing, SRM, FDR…).  
Construit avec **Gradio + FAISS + BM25** sur un corpus Wikipédia curaté.  
L’index FAISS est pré-calculé et chargé automatiquement au démarrage de la Space.

---

## ⚙️ Installation locale

```bash
# 1. Créer et activer un environnement virtuel
python3 -m venv .venv && source .venv/bin/activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l’application Gradio
python app.py
```

---

## 🗂 Structure du projet

| Fichier                         | Description                                                                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_open_dataset_curated.py` | Récupère et nettoie des pages Wikipédia FR/EN sur A/B testing, SRM, etc., puis publie le dataset `lmhdii/experiment-brief-open` sur Hugging Face. |
| `index_open_faiss.py`           | Crée l’index vectoriel FAISS (embeddings `sentence-transformers`) pour la recherche sémantique.                                                   |
| `hybrid_search.py`              | Combine FAISS (dense) et BM25 (sparse) pour tester la recherche en ligne de commande.                                                             |
| `app.py`                        | Interface utilisateur web (Gradio). Permet la recherche par question, langue et nombre de passages.                                               |
| `hf-space/`                     | Version simplifiée utilisée pour le déploiement sur Hugging Face Spaces (sans les fichiers volumineux).                                           |
| `requirements.txt`              | Liste des dépendances Python.                                                                                                                     |

---

## 📚 Dataset

**Nom :** [`lmhdii/experiment-brief-open`](https://huggingface.co/datasets/lmhdii/experiment-brief-open)
**Langues :** 🇫🇷 / 🇬🇧
**Schéma :**

```
id, source_type, title, url, language, year, topics, text
```

Corpus issu de Wikipédia, filtré pour les thématiques :

> A/B testing, SRM, CUPED, séquentiel, guardrails, etc.

---

## 🚀 Déploiement

* **Hébergé sur Hugging Face Spaces (CPU normal)**
* **Index FAISS** stocké via **Git LFS**
* Chargement automatique de l’index au démarrage pour des requêtes instantanées

---

## 🧭 Fonctionnalités principales

✅ Recherche hybride (FAISS + BM25)
✅ Bilingue (FR / EN)
✅ Interface Gradio légère et claire
✅ Dataset public et reproductible
✅ Index vectoriel persistant sur HF Space

---

## 🧩 Roadmap (prochaines améliorations)

* 🔹 **Score de confiance** (affichage du cosine similarity)
* 🔹 **Filtres dynamiques** (FR / EN)
* 🔹 **Exportation des résultats** (JSON / CSV)
* 🔹 **Dockerfile** pour exécution locale reproductible
* 🔹 **Interface améliorée** (cartes + barres de confiance)

---

## 🧠 Crédits

Projet réalisé par **El Mehdi BELAHNECH** dans le cadre de la formation *DataBird – Data Science & IA (2025)*.
Encadré par [DataBird](https://www.databird.ai/) & hébergé sur [Hugging Face](https://huggingface.co/).

---

## 📄 Licence

Ce projet est distribué sous licence **MIT** — utilisation libre et attribution souhaitée.

---

> 💡 *Pour les curieux : les fichiers FAISS volumineux sont stockés via Git LFS et ignorés du dépôt GitHub principal.*

```