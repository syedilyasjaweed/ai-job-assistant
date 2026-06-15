"""
Phase 2: Company Research Agent
Uses Claude's web search tool to research a company and produce a
structured profile (mission, recent news, tech stack, culture notes)
for use in Phase 3's cover letter generator.

Usage:
    python phase2_company_research.py "Company Name"
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CACHE_DIR = Path("data/company_research_cache")
CACHE_TTL_DAYS = 7  # refresh cache after this many days

MODEL = "claude-sonnet-4-6"

client = Anthropic(api_key=ANTHROPIC_API_KEY)


def _cache_path(company_name: str) -> Path:
    """Create a safe filename for the cache based on the company name."""
    safe_name = hashlib.md5(company_name.lower().encode()).hexdigest()[:10]
    return CACHE_DIR / f"{safe_name}.json"


def _load_from_cache(company_name: str) -> dict | None:
    path = _cache_path(company_name)
    if not path.exists():
        return None

    with open(path, "r") as f:
        cached = json.load(f)

    cached_time = datetime.fromisoformat(cached["_cached_at"])
    if datetime.now() - cached_time > timedelta(days=CACHE_TTL_DAYS):
        return None  # cache expired

    return cached


def _save_to_cache(company_name: str, profile: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    profile["_cached_at"] = datetime.now().isoformat()
    profile["_company_name"] = company_name

    with open(_cache_path(company_name), "w") as f:
        json.dump(profile, f, indent=2)


def research_company(company_name: str, force_refresh: bool = False) -> dict:
    """
    Research a company using Claude's web search tool and return
    a structured profile dict.
    """
    if not force_refresh:
        cached = _load_from_cache(company_name)
        if cached:
            print(f"[cache] Using cached profile for {company_name}")
            return cached

    print(f"[search] Researching {company_name}...")

    prompt = f"""Research the company "{company_name}" using web search.

Find and summarize:
1. Mission / values - what they say their purpose and values are
2. Recent news - 2-3 notable recent developments (product launches, funding, etc.)
3. Tech stack - technologies, languages, or platforms they use or are known for
4. Culture notes - anything about work culture, remote policy, or what they value in employees

Respond ONLY with a JSON object in this exact format, no preamble, no markdown:
{{
  "mission": "...",
  "recent_news": ["...", "..."],
  "tech_stack": ["...", "..."],
  "culture_notes": "..."
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    # Collect all text blocks (Claude may interleave text and search results)
    text_parts = [
        block.text for block in response.content if block.type == "text"
    ]
    full_text = "\n".join(text_parts).strip()

    # Strip markdown code fences if present
    if full_text.startswith("```"):
        full_text = full_text.strip("`")
        if full_text.startswith("json"):
            full_text = full_text[4:].strip()

    try:
        profile = json.loads(full_text)
    except json.JSONDecodeError:
        print("[warn] Could not parse JSON response. Raw output:")
        print(full_text)
        profile = {
            "mission": "",
            "recent_news": [],
            "tech_stack": [],
            "culture_notes": "",
            "_raw_response": full_text,
        }

    _save_to_cache(company_name, profile)
    return profile


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase2_company_research.py \"Company Name\"")
        sys.exit(1)

    company_name = sys.argv[1]
    force_refresh = "--refresh" in sys.argv

    profile = research_company(company_name, force_refresh=force_refresh)

    print("\n=== Company Profile ===")
    print(f"Company: {company_name}")
    print(f"\nMission:\n{profile.get('mission', 'N/A')}")
    print(f"\nRecent News:")
    for item in profile.get("recent_news", []):
        print(f"  - {item}")
    print(f"\nTech Stack: {', '.join(profile.get('tech_stack', []))}")
    print(f"\nCulture Notes:\n{profile.get('culture_notes', 'N/A')}")


if __name__ == "__main__":
    main()