## AI Job Application Assistant

An AI-powered tool that helps tailor job applications using a RAG pipeline.

## What It Does
- Matches your resume bullets to any job description using semantic search
- Researches companies automatically using Claude's web search
- (Coming soon) Generates tailored cover letters

## Pipeline Overview

### Phase 1 — Resume Matcher
- Embeds 18 resume bullets using VoyageAI (`voyage-3-large`)
- Stores vectors in Pinecone (`resume-bullets` index)
- For any job description, returns top 5 matching bullets with similarity scores

### Phase 2 — Company Research Agent
- Takes a company name as input
- Uses Claude (`claude-sonnet-4-6`) with web search to find mission, recent news, tech stack, and culture notes
- Caches results locally for 7 days to avoid redundant API calls
- Supports `--refresh` flag to force a fresh search

## Phase 3 — Cover Letter Generator

- Combines Phase 1 bullet matches + Phase 2 company profile
- Generates a tailored cover letter via Claude (`claude-sonnet-4-6`)
- Saves output to `data/cover_letters/` (gitignored — contains personal application content)
## Setup

pip install -r requirements.txt

Create a `.env` file:

ANTHROPIC_API_KEY=your_key
VOYAGE_API_KEY=your_key
PINECONE_API_KEY=your_key

## Usage

# Phase 1 - match resume to job description
python phase1_matcher.py

# Phase 2 - research a company
python phase2_company_research.py "Anthropic"

# Phase 3 — generate a tailored cover letter
python phase3_cover_letter.py --company "Anthropic" --role "AI Engineer" --jd "paste job description here"

# Force a fresh company research lookup
python phase3_cover_letter.py --company "Anthropic" --role "AI Engineer" --jd "..." --refresh

# Force fresh search (bypass cache)
python phase2_company_research.py "Anthropic" --refresh

## Tech Stack
- Python, Anthropic Claude API, VoyageAI, Pinecone
- RAG pipeline: embeddings → vector search → semantic matching