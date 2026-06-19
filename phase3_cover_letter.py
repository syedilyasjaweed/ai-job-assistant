"""
Phase 3 - Cover Letter Generator
Combines Phase 1 resume bullet matches + Phase 2 company research
to generate a tailored cover letter via Claude.

Usage:
    python phase3_cover_letter.py --company "Anthropic" --role "AI Engineer" --jd "path/to/jd.txt"
    python phase3_cover_letter.py --company "Anthropic" --role "AI Engineer" --jd "We are looking for..."
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import anthropic
from dotenv import load_dotenv

# Import Phase 1 functions directly
from phase1_matcher import embed_query, search_pinecone

# Import Phase 2 function directly
from phase2_company_research import research_company

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL          = "claude-sonnet-4-6"
COVER_LETTER_DIR = Path("data/cover_letters")

# ── System prompt for Claude ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert career coach and professional writer
specializing in AI/ML and tech roles. You write compelling, authentic cover
letters that:

- Open with a specific hook tied to the company's mission or recent work
- Weave in the candidate's exact experience naturally (don't just list bullets)
- Connect the candidate's background to the company's specific needs
- Stay under 400 words (3-4 paragraphs)
- Sound human and confident, never generic or robotic
- Use professional but conversational tone - no fluff, no buzzwords

Formatting rules:
- Do NOT include a salutation (no Dear Hiring Manager)
- Do NOT include a subject line or date
- Start directly with the opening paragraph
- End after the closing paragraph - no signature block
- Plain text only, no markdown formatting"""


# ── Step 1: Get matched bullets from Phase 1 ─────────────────────────────────
def get_matched_bullets(jd_text):
    print("\n[1/3] Running Phase 1 resume matcher...")
    query_embedding = embed_query(jd_text)
    matches         = search_pinecone(query_embedding)

    bullets = []
    for match in matches:
        bullets.append({
            "bullet": match["metadata"]["text"],
            "score":  match["score"]
        })

    print(f"      ✓ {len(bullets)} bullets matched")
    return bullets


# ── Step 2: Get company profile from Phase 2 ─────────────────────────────────
def get_company_profile(company, refresh):
    print(f"\n[2/3] Running Phase 2 company research for '{company}'...")
    profile = research_company(company, force_refresh=refresh)
    print(f"      ✓ Profile loaded")
    return profile


# ── Step 3: Build the prompt and call Claude ──────────────────────────────────
def generate_cover_letter(company, role, jd_text, bullets, profile):
    print(f"\n[3/3] Generating cover letter via Claude ({MODEL})...")

    # Format bullets for the prompt
    bullets_text = "\n".join(
        f"  {i+1}. [score: {b['score']:.2f}] {b['bullet']}"
        for i, b in enumerate(bullets)
    )

    # Format company profile for the prompt
    profile_text = json.dumps(profile, indent=2)

    user_prompt = f"""Write a tailored cover letter for this job application.

== TARGET ROLE ==
Company: {company}
Position: {role}

== JOB DESCRIPTION ==
{jd_text[:3000]}

== CANDIDATE'S TOP MATCHING RESUME BULLETS ==
{bullets_text}

== COMPANY RESEARCH PROFILE ==
{profile_text}

Instructions:
- Open by referencing something specific from the company profile
- Naturally work in 2-3 of the top resume bullets as proof of fit
- The candidate has an MS in Information Technology Management from
  Illinois Institute of Technology and hands-on AI/ML project experience
- Close with enthusiasm and a clear call to action
- Plain text only, under 400 words"""

    client   = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )

    letter = response.content[0].text.strip()
    print(f"      ✓ Cover letter generated ({len(letter.split())} words)")
    return letter


# ── Save to file ──────────────────────────────────────────────────────────────
def save_cover_letter(company, role, letter):
    COVER_LETTER_DIR.mkdir(parents=True, exist_ok=True)

    safe_company = re.sub(r"[^\w\-]", "_", company.lower())
    safe_role    = re.sub(r"[^\w\-]", "_", role.lower().replace(" ", "_"))
    date_str     = datetime.now().strftime("%Y%m%d")
    filename     = f"{safe_company}_{safe_role}_{date_str}.txt"

    output_path  = COVER_LETTER_DIR / filename
    output_path.write_text(letter, encoding="utf-8")
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 3 - Cover Letter Generator")
    parser.add_argument("--company", required=True, help="Target company name")
    parser.add_argument("--role",    required=True, help="Target job title")
    parser.add_argument("--jd",      required=True, help="Job description: file path or raw text")
    parser.add_argument("--refresh", action="store_true", help="Force refresh company research cache")
    args = parser.parse_args()

    # Load JD — accept either a file path or raw text
    jd_path = Path(args.jd)
    jd_text = jd_path.read_text(encoding="utf-8").strip() if jd_path.exists() else args.jd.strip()

    # Run all three steps
    bullets = get_matched_bullets(jd_text)
    profile = get_company_profile(args.company, args.refresh)
    letter  = generate_cover_letter(args.company, args.role, jd_text, bullets, profile)

    # Save and print
    output_path = save_cover_letter(args.company, args.role, letter)

    print(f"\n{'='*60}")
    print(f"  COVER LETTER — {args.role} @ {args.company}")
    print(f"{'='*60}\n")
    print(letter)
    print(f"\n{'='*60}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()