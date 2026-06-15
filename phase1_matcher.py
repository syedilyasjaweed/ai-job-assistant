import os
from dotenv import load_dotenv
import voyageai
from pinecone import Pinecone

load_dotenv()

VOYAGE_API_KEY   = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = "resume-bullets"
TOP_K            = 5

def embed_query(query_text):
    client = voyageai.Client(api_key=VOYAGE_API_KEY)
    result = client.embed(
        texts=[query_text],
        model="voyage-3-large",
        input_type="query"
    )
    return result.embeddings[0]

def search_pinecone(query_embedding):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True
    )
    return results["matches"]

def display_results(matches, job_description):
    print("\n" + "=" * 60)
    print("  TOP MATCHING RESUME BULLETS")
    print("=" * 60)
    print(f"\nJob description: \"{job_description[:80].strip()}...\"\n")
    for rank, match in enumerate(matches, start=1):
        score  = match["score"]
        bullet = match["metadata"]["text"]
        bar    = "█" * int(score * 20)
        print(f"  #{rank}  Score: {score:.3f}  {bar}")
        print(f"       {bullet}")
        print()
    print("=" * 60)
    print("\n💡 Copy the highest scoring bullets into your application\n")

def get_job_description():
    print("\n📋 Paste the job description below.")
    print("   Press Enter twice when done.")
    print("-" * 60)
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()

def main():
    print("=" * 60)
    print("  AI Job Assistant — Resume Matcher")
    print("=" * 60)

    if not VOYAGE_API_KEY:
        raise ValueError("❌ VOYAGE_API_KEY missing from .env")
    if not PINECONE_API_KEY:
        raise ValueError("❌ PINECONE_API_KEY missing from .env")

    job_description = get_job_description()

    if not job_description:
        print("❌ No job description entered. Exiting.")
        return

    print("\n⏳ Embedding job description...")
    query_embedding = embed_query(job_description)

    print("🔍 Searching Pinecone for matching bullets...")
    matches = search_pinecone(query_embedding)

    display_results(matches, job_description)

    while True:
        again = input("Search another job? (y/n): ").strip().lower()
        if again != "y":
            break
        job_description = get_job_description()
        print("\n⏳ Embedding job description...")
        query_embedding = embed_query(job_description)
        print("🔍 Searching Pinecone...")
        matches = search_pinecone(query_embedding)
        display_results(matches, job_description)

    print("\n✅ Done. Happy job hunting!\n")

if __name__ == "__main__":
    main()
