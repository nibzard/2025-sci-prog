# check_chromadb.py
import chromadb
import os

def check_all_collections():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    chroma_dir = os.path.join(project_root, "chroma")

    print("="*60)
    print("ChromaDB Diagnostic Report")
    print(f"Using ChromaDB directory: {chroma_dir}")
    print("="*60)

    client = chromadb.PersistentClient(path=chroma_dir)

    collections = client.list_collections()
    print(f"\nTotal collections: {len(collections)}")

    for col in collections:
        print(f"\n📁 Collection: {col.name}")
        print(f"   Documents: {col.count()}")

        if col.count() > 0:
            result = col.get(limit=1)
            print(f"   Sample IDs: {result['ids']}")
            print(f"   Sample metadata: {result['metadatas']}")

    print("\n" + "="*60)

if __name__ == "__main__":
    check_all_collections()
