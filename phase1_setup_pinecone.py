import os
import time
from dotenv import load_dotenv
import voyageai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

VOYAGE_API_KEY   = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = "resume-bullets"

def load_bullets(filepath):
    bullets = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                bullets.append(line)
    print(f"✅ Loaded {len(bullets)} resume bullets")
    return bullets

def embed_bullets(bullets):
    client = voyageai.Client(api_key=VOYAGE_API_KEY)
    print(f"⏳ Embedding {len(bullets)} bullets with VoyageAI...")
    result = client.embed(
        texts=bullets,
        model="voyage-3-large",
        input_type="document"
    )
    embeddings = result.embeddings
    print(f"✅ Got {len(embeddings)} embeddings, dimension = {len(embeddings[0])}")
    return embeddings

def upload_to_pinecone(bullets, embeddings):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"⏳ Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=len(embeddings[0]),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("⏳ Waiting for index to be ready...")
        while True:
            status = pc.describe_index(INDEX_NAME).status
            if status.get("ready"):
                break
            time.sleep(2)
        print(f"✅ Index '{INDEX_NAME}' is ready")
    else:
        print(f"✅ Index '{INDEX_NAME}' already exists")

    vectors = []
    for i, (bullet, embedding) in enumerate(zip(bullets, embeddings)):
        vectors.append({
            "id":       f"bullet-{i}",
            "values":   embedding,
            "metadata": {"text": bullet}
        })

    index = pc.Index(INDEX_NAME)
    index.upsert(vectors=vectors)
    print(f"✅ Uploaded {len(vectors)} vectors to Pinecone")

    time.sleep(2)
    stats = index.describe_index_stats()
    print(f"✅ Pinecone index now has {stats['total_vector_count']} vectors")

def main():
    print("=" * 50)
    print("  Phase 1 Setup — uploading resume to Pinecone")
    print("=" * 50)

    if not VOYAGE_API_KEY:
        raise ValueError("❌ VOYAGE_API_KEY missing from .env")
    if not PINECONE_API_KEY:
        raise ValueError("❌ PINECONE_API_KEY missing from .env")

    bullets    = load_bullets("data/resume_bullets.txt")
    embeddings = embed_bullets(bullets)
    upload_to_pinecone(bullets, embeddings)

    print("\n🎉 Done! Now run: python phase1_matcher.py")

if __name__ == "__main__":
    main()
