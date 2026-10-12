# Drug Reviews — Unmet Clinical Need Analysis

A pipeline to identify medical conditions where, despite available treatments, patient
reviews suggest remaining unmet clinical need — using the UCI Drug Reviews
(Druglib.com) dataset as a working example.

## What this does

1. Clusters near-duplicate condition names (e.g. "migraine" vs "migraines") using
   sentence embeddings and agglomerative clustering.
2. Ranks conditions by review volume, average rating, and share of low ratings to
   surface candidates with the most signal.
3. For the selected condition (depression, in this run), analyses competitor drugs,
   and extracts recurring side-effect/benefit themes via three complementary methods:
   word-frequency clouds, BERTopic on raw reviews, and BERTopic on LLM-normalised
   review topics.

See the notebook's Discussion and Executive Summary sections for the actual findings.

## Setup

```bash
pip install -r requirements.txt
mkdir data
```

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```
Then open `notebook-drug-reviews.ipynb` and run top to bottom.

## Runtime

Most of the notebook runs in minutes. The per-review LLM topic extraction step
(local Phi-3.5-mini-instruct, run on CPU to avoid sending data to a third party)
took roughly 3 hours on ~300 reviews -> worth knowing before re-running the full
notebook.

## Data

This project uses the [Drug Reviews](https://doi.org/10.24432/C55G6J) dataset by Surya Kallumadi and Felix Gräßer,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with additional terms: research use
only, no commercial use, and no redistribution of the data itself.

Kallumadi, S. & Gräßer, F. (2018). Drug Reviews (Druglib.com) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55G6J.

The dataset is fetched at runtime via ucimlrepo, never committed to this repo.
data/ holds only locally-generated derivatives (embeddings, cached model outputs,
figures) and is gitignored, nothing in it is redistributed. Only aggregated results
(counts, percentages, cluster keyword representations across groups of at least 10
reviews) are shown in the notebook or report; no individual review text or per-review
LLM output is published.

## Structure

* notebook-drug-reviews.ipynb -> full analysis
* src/functions.py -> reusable functions (tokenisation, word clouds, topic modelling)
* data/ -> local-only, gitignored derived artifacts
* report/ -> local-only, gitignored final write-up

## Tools

Python, pandas, NumPy, scikit-learn, NLTK, WordCloud, PyTorch, Hugging Face
Transformers, sentence-transformers, BERTopic, UMAP, HDBSCAN.

## Note

Structure and content reflect my own analysis and interpretation. I used an AI coding
assistant (Claude Code) to help organise the codebase into reusable modules, troubleshoot
environment/dependency issues, and review code for bugs. The assistant worked from code
and file structure only, never executing cells or scripts that would surface actual
review content, and at no point was raw dataset text read, processed, or transmitted to
it, consistent with the dataset's no-redistribution terms.


