# AI Job Application Assistant

An AI-powered tool that helps tailor job applications by semantically matching resume bullet points to job descriptions — built using a Retrieval-Augmented Generation (RAG) pipeline with VoyageAI embeddings, Pinecone vector search, and the Anthropic Claude API.

## Overview

Job searching often means manually rewriting your resume for every application — guessing which bullet points are most relevant to a given role. This project automates that step.

Paste in any job description, and the tool returns the top 5 resume bullets that are the strongest semantic match — ranked by similarity score — so you know exactly what to highlight in your application.

**Status:** Phase 1 complete (resume matcher). Phases 2 and 3 (company research agent and cover letter generator) are in progress — see [Roadmap](#roadmap).

## How It Works

This project implements a full RAG (Retrieval-Augmented Generation) pipeline:

### Indexing (run once)
1. **Chunking** — Resume bullets are stored one per line, with each bullet treated as a single self-contained chunk (manual chunking — ideal since each bullet is already one complete thought).
2. **Embedding** — Each bullet is converted into a 1024-dimension vector using VoyageAI's `voyage-3-large` model (`input_type="document"`).
3. **Storage** — All vectors are uploaded to a Pinecone vector database index, with the original bullet text stored as metadata for retrieval.

### Retrieval (run for every job application)
4. **Query embedding** — The pasted job description is embedded using the same VoyageAI model, but with `input_type="query"` — VoyageAI optimizes query and document embeddings differently to improve match accuracy.
5. **Semantic search** — The query vector is compared against all stored resume bullet vectors using cosine similarity, returning the top 5 closest matches (`top_k=5`).
6. **Results** — Matches are returned with similarity scores (0–1 scale). Scores in the 0.6–0.7 range indicate a good match.

### Generation (Phase 3 — planned)
7. The top-matched bullets and job description will be passed to the Claude API to automatically generate a tailored cover letter, completing the full retrieve → augment → generate loop.

## Tech Stack

- **VoyageAI** (`voyage-3-large`) — text embeddings
- **Pinecone** — vector database for semantic search
- **Anthropic Claude API** — generation (Phase 3)
- **Python** — `anthropic`, `voyageai`, `pinecone`, `python-dotenv`

## Project Structure

```
ai-job-assistant/
├── .env                        # API keys (not committed)
├── .gitignore
├── requirements.txt
├── data/
│   └── resume_bullets.txt      # Resume bullets, one per line
├── phase1_setup_pinecone.py    # Run once: embeds and uploads bullets to Pinecone
└── phase1_matcher.py           # Run per job: paste JD, get top 5 matching bullets
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   ANTHROPIC_API_KEY=your_key_here
   VOYAGE_API_KEY=your_key_here
   PINECONE_API_KEY=your_key_here
   ```

3. Add your resume bullets to `data/resume_bullets.txt` (one bullet per line).

4. Upload your resume bullets to Pinecone (run once, or whenever bullets change):
   ```bash
   python3 phase1_setup_pinecone.py
   ```

5. Run the matcher for any job application:
   ```bash
   python3 phase1_matcher.py
   ```
   Paste the job description when prompted, and get your top 5 matching bullets with similarity scores.

## Roadmap

- [x] **Phase 1** — Resume matcher (semantic search with VoyageAI + Pinecone)
- [ ] **Phase 2** — Company research agent using Claude's web search tool
- [ ] **Phase 3** — Cover letter generator using LangChain prompt chaining
