# Ask Your PDFs

A Python RAG application that ingests a folder of PDFs and answers questions with page-level
citations in the form `[filename, p. N]`.

## How it works

| Step | Implementation |
|---|---|
| PDF to text | `pypdf`, one page at a time |
| Chunking | 800 characters with 150 characters of overlap, split within a page on sentence or word boundaries |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, normalized |
| Vector store | Chroma (persistent, cosine distance), with `filename` and `page` as chunk metadata |
| Retrieval | Top-k semantic search (k = 5 by default, adjustable in the UI) |
| Generation | Google Gemini answers from the retrieved chunks only |
| UI | Streamlit |

Chunks never cross a page boundary, so every chunk belongs to exactly one page and its citation is
exact. `N` is the page's position in the PDF file, starting at 1.

### When the answer is not in the documents

The app replies `I don't know based on the provided documents.` in three situations:

1. The best retrieved chunk has a cosine similarity below `0.30`. The LLM is not called. On the
   sample corpus, unrelated questions scored at most 0.21 and answerable ones at least 0.50.
2. The chunks are related but do not contain the answer. The system prompt tells Gemini to reply
   with this exact sentence.
3. The answer has no citation pointing at a retrieved page. Citations to pages that were not
   retrieved are removed, and an answer left without any valid citation is rejected.

## Setup

Requires Python 3.14 (the version the pinned dependencies were tested on) and a
[Gemini API key](https://aistudio.google.com/apikey).

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Linux/macOS: cp .env.example .env
```

Put your key into `.env` as `GEMINI_API_KEY`. The default model is `gemini-3.5-flash-lite`; set
`GEMINI_MODEL` to use another one.

## Usage

Build the vector database. This downloads the embedding model on the first run.

```
python ingest.py
```

To index a different folder, run `python ingest.py --pdf-dir path/to/pdfs`. Each run replaces the
previous index.

Start the app:

```
streamlit run app.py
```

## Sample questions and expected citations

| Question | Expected answer | Expected citation |
|---|---|---|
| How many identical layers does the Transformer encoder have? | 6 | `[attention_is_all_you_need.pdf, p. 3]` |
| What dropout rate was used when training the Transformer? | 0.1 | `[attention_is_all_you_need.pdf, p. 8]` |
| What percentage of input tokens does BERT mask for masked language modeling? | 15% | `[bert.pdf, p. 4]` |
| Which model does RAG use as its generator? | BART | `[rag_lewis_et_al.pdf, p. 3]` |
| Which encoders does Dense Passage Retriever use for questions and passages? | Two independent BERT networks | `[dense_passage_retrieval.pdf, p. 3]` |
| Which concatenation of sentence embeddings does SBERT use for its classification objective? | (u, v, \|u-v\|) | `[sentence_bert.pdf, p. 7]` |
| What is the capital of Australia? | I don't know based on the provided documents. | none |

## Sample documents

The PDFs in `data/pdfs` are publicly available papers from arXiv:

| File | Paper |
|---|---|
| `attention_is_all_you_need.pdf` | Vaswani et al., *Attention Is All You Need*, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| `bert.pdf` | Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers*, [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) |
| `dense_passage_retrieval.pdf` | Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) |
| `rag_lewis_et_al.pdf` | Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| `sentence_bert.pdf` | Reimers and Gurevych, *Sentence-BERT*, [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) |
