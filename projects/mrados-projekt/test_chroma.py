import chromadb


def test_chroma():
    client = chromadb.Client(
        chromadb.config.Settings(persist_directory="./chroma")
    )
    collection = client.get_collection("laptop_knowledge")

    results = collection.query(
        query_texts=["Legion Y540"],
        n_results=3
    )

    print(f"results: {results['documents']}")

def main() -> None:
    test_chroma()
    

if __name__ == "__main__":
    main()