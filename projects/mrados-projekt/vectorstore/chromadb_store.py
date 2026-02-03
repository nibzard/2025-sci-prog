# vectorstore/chromadb_store.py
"""
ChromaDB Storage Module - Embeds and stores markdown files in ChromaDB.
Supports organizing documents by laptop for easier retrieval.
"""

import chromadb
from chromadb.utils import embedding_functions
import os
import re

print("ChromaDB store module loaded.")


def get_chroma_client(chroma_dir: str | None = None) -> chromadb.PersistentClient:
    """
    Get a persistent ChromaDB client.
    
    Args:
        chroma_dir: Directory for ChromaDB storage (default: project_root/chroma)
        
    Returns:
        PersistentClient instance
    """
    if chroma_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        chroma_dir = os.path.join(project_root, "chroma")
    
    print(f"Using ChromaDB directory: {chroma_dir}")
    return chromadb.PersistentClient(path=chroma_dir)


def store_markdown_in_chroma(
    md_path: str, 
    collection_name: str = "laptop_knowledge",
    laptop_name: str | None = None
) -> bool:
    """
    Store markdown file in ChromaDB with embeddings.
    
    Args:
        md_path: Path to the markdown file (relative or absolute)
        collection_name: Name of the ChromaDB collection
        laptop_name: Optional laptop name for metadata
        
    Returns:
        True if successful, False otherwise
    """
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    chroma_dir = os.path.join(project_root, "chroma")
    
    # Handle both relative and absolute paths
    if os.path.isabs(md_path):
        md_full_path = md_path
    else:
        md_full_path = os.path.join(project_root, md_path)
    
    if not os.path.exists(md_full_path):
        print(f"ERROR: File not found: {md_full_path}")
        return False
    
    # Create persistent client
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Set up embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # Read markdown file
    with open(md_full_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    print(f"Read {len(content)} characters from {md_path}")

    # Generate unique ID from file path
    doc_id = f"reddit_{os.path.basename(md_path).replace('.md', '')}"
    # Clean up the ID
    doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
    
    # Extract laptop name from path if not provided
    if laptop_name is None:
        # Try to extract from path like knowledge/reddit/lenovo_legion_y540/...
        path_parts = md_path.replace("\\", "/").split("/")
        for i, part in enumerate(path_parts):
            if part == "reddit" and i + 1 < len(path_parts):
                laptop_name = path_parts[i + 1]
                break
    
    # Prepare metadata
    metadata = {
        "source": md_path,
        "type": "reddit",
    }
    if laptop_name:
        metadata["laptop"] = laptop_name
    
    # Check if already exists
    existing = collection.get(ids=[doc_id])
    if existing['ids']:
        print(f"Document {doc_id} already exists. Updating...")
        collection.update(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata]
        )
    else:
        print(f"Adding new document {doc_id}...")
        collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )

    print(f"Stored in ChromaDB: {md_path}")
    print(f"Collection '{collection_name}' now has {collection.count()} documents")
    
    return True


def store_all_markdown_for_laptop(
    laptop_name: str,
    knowledge_dir: str = "knowledge/reddit",
    collection_name: str = "laptop_knowledge"
) -> int:
    """
    Store all markdown files for a specific laptop.
    
    Args:
        laptop_name: Name of the laptop
        knowledge_dir: Base directory for markdown files
        collection_name: ChromaDB collection name
        
    Returns:
        Number of files stored
    """
    # Import here to avoid circular import
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import get_laptop_markdown_dir
    
    laptop_dir = get_laptop_markdown_dir(laptop_name, knowledge_dir)
    
    if not os.path.exists(laptop_dir):
        print(f"No markdown directory found for {laptop_name}")
        return 0
    
    stored_count = 0
    for filename in os.listdir(laptop_dir):
        if filename.endswith('.md'):
            filepath = os.path.join(laptop_dir, filename)
            if store_markdown_in_chroma(filepath, collection_name, laptop_name):
                stored_count += 1
    
    print(f"\nStored {stored_count} documents for {laptop_name}")
    return stored_count


def query_collection(
    query: str,
    collection_name: str = "laptop_knowledge",
    n_results: int = 5,
    laptop_filter: str | None = None
) -> list[str]:
    """
    Query the ChromaDB collection.
    
    Args:
        query: Search query
        collection_name: Collection to search
        n_results: Number of results to return
        laptop_filter: Optional filter by laptop name
        
    Returns:
        List of matching document contents
    """
    client = get_chroma_client()
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    
    # Build query
    where_filter = None
    if laptop_filter:
        where_filter = {"laptop": laptop_filter}
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )
    
    return results['documents'][0] if results['documents'] else []


def main() -> None:
    """Test storing markdown files."""
    
    # Store the Legion Y540 review
    store_markdown_in_chroma(
        "knowledge/reddit/my-legion-y540-still-serving-me-after-5-years.md",
        collection_name="laptop_knowledge"
    )
    
    print("\n" + "="*60)
    print("Storage complete!")
    print("="*60)


if __name__ == "__main__":
    main()
