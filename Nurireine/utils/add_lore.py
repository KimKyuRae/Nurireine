import sys
import os
import time
from pathlib import Path

# Add project root to sys.path to allow imports from Nurireine
# Current file: project_root/Nurireine/utils/add_lore.py
# Root dir: project_root/
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

try:
    import chromadb
    from Nurireine.ai.embeddings import GGUFEmbeddingFunction
    from Nurireine import config
except ImportError as e:
    print(f"Error: Required modules not found. Ensure you are running from the project root or have dependencies installed.")
    print(f"Import error: {e}")
    sys.exit(1)

def add_lore(content: str):
    """
    Manually add lore content to the L3 Vector Database (ChromaDB).
    """
    print("Initializing ChromaDB connection...")
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
        embedding_fn = GGUFEmbeddingFunction()
        
        collection = client.get_or_create_collection(
            name=config.memory.collection_name,
            embedding_function=embedding_fn
        )
    except Exception as e:
        print(f"Failed to connect to ChromaDB: {e}")
        return

    # Duplicate check
    print("Checking for existing similar lore...")
    threshold = config.memory.l3_similarity_threshold
    try:
        existing = collection.query(
            query_texts=[content],
            n_results=1,
            where={"context": "lore"}
        )
        
        if existing['distances'] and len(existing['distances'][0]) > 0:
            distance = existing['distances'][0][0]
            if distance < threshold:
                print(f"\n[Warning] Similar lore already exists (Distance: {distance:.4f})")
                print(f"Existing content: \"{existing['documents'][0][0]}\"")
                confirm = input("\nDo you still want to add this new lore? (y/N): ")
                if confirm.lower() != 'y':
                    print("Operation cancelled.")
                    return
    except Exception as e:
        print(f"Duplicate check failed (skipping): {e}")

    # Prepare data
    timestamp_ms = int(time.time() * 1000)
    fact_id = f"lore_manual_{timestamp_ms}"
    
    metadata = {
        "context": "lore",
        "timestamp": time.time(),
        "topic": "manual_lore",
        "keywords": "manual,lore"
    }
    
    # Add to collection
    print("Adding lore to L3 memory...")
    try:
        collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[fact_id]
        )
        print("\n[Success] Lore added successfully!")
        print(f"ID: {fact_id}")
        print(f"Content: {content}")
    except Exception as e:
        print(f"Failed to add lore: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Nurireine/utils/add_lore.py \"Your lore content here\"")
        print("Example: python Nurireine/utils/add_lore.py \"누리레느는 민트 초코를 좋아한다.\"")
        sys.exit(1)
    
    lore_text = " ".join(sys.argv[1:])
    add_lore(lore_text)
