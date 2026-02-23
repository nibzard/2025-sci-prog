import chromadb
from chromadb.utils import embedding_functions
from google import genai
import json
import os
import sys
from dotenv import load_dotenv
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_analysis_output_path, laptop_name_to_slug

# Load environment variables
load_dotenv()

print("LLM Analysis module loaded.")


class LaptopAnalyzer:
    """Analyzes laptop information using ChromaDB and Google Gemini."""
    
    def __init__(self, collection_name: str = "laptop_knowledge"):
        """Initialize the analyzer with ChromaDB and Gemini."""
        
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client_gemini = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"

        
        # Initialize ChromaDB client
        # self.client = chromadb.Client(
        #     chromadb.config.Settings(
        #         persist_directory="./chroma"
        #     )
        # )

        # Initialize ChromaDB client (MUST match storage)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        chroma_dir = os.path.join(project_root, "chroma")

        print(f"Using ChromaDB directory: {chroma_dir}")

        self.client = chromadb.PersistentClient(path=chroma_dir)

        
        # Set up embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        print(f"✓ Connected to ChromaDB collection: {collection_name}")
        print(f"✓ Documents in collection: {self.collection.count()}")
    
    
    def query_chromadb(self, laptop_query: str, n_results: int = 5) -> List[str]:
        """
        Query ChromaDB for relevant laptop information.
        
        Args:
            laptop_query: Search query (e.g., "Lenovo Legion Y540")
            n_results: Number of results to retrieve
            
        Returns:
            List of relevant document chunks
        """
        results = self.collection.query(
            query_texts=[laptop_query],
            n_results=n_results
        )
        
        documents = results['documents'][0] if results['documents'] else []
        print(f"✓ Retrieved {len(documents)} relevant documents from ChromaDB")
        
        return documents
    
    
    def analyze_laptop(self, laptop_name: str, context_docs: List[str]) -> Dict:
        """
        Analyze laptop using Gemini based on ChromaDB context.
        
        Args:
            laptop_name: Name of the laptop
            context_docs: Relevant documents from ChromaDB
            
        Returns:
            Structured analysis with pros, cons, and sentiment score
        """
        
        # Combine context documents
        context = "\n\n---\n\n".join(context_docs)
        
        # Create prompt for Gemini
        prompt = f"""
You are an expert laptop reviewer analyzing Reddit discussions and user feedback.

LAPTOP: {laptop_name}

CONTEXT FROM REDDIT DISCUSSIONS:
{context}

Based on the above Reddit discussions and user feedback, provide a comprehensive analysis in the following JSON format:

{{
    "laptop_name": "{laptop_name}",
    "pros": [
        "List 3-5 key advantages mentioned by users",
        "Focus on real-world user experiences",
        "Include specific details from the context"
    ],
    "cons": [
        "List 3-5 key disadvantages mentioned by users",
        "Focus on common complaints or issues",
        "Include specific details from the context"
    ],
    "sentiment_score": 75,
    "sentiment_explanation": "Brief explanation of why this score was given (1-2 sentences)",
    "key_themes": [
        "List 2-3 recurring themes from user discussions",
        "E.g., 'Long-term reliability', 'Thermal performance', 'Value for money'"
    ],
    "user_recommendation": "Overall recommendation based on user sentiment (1-2 sentences)"
}}

SCORING GUIDELINES:
- sentiment_score: 1-100 scale where:
  - 90-100: Overwhelmingly positive, highly recommended
  - 75-89: Mostly positive with minor issues
  - 60-74: Mixed reviews, has notable pros and cons
  - 40-59: More negative than positive
  - 1-39: Mostly negative, not recommended

IMPORTANT:
- Base your analysis ONLY on the provided context
- If context is insufficient, mention this in the analysis
- Be specific and cite examples from the Reddit discussions
- Return ONLY valid JSON, no additional text
"""
        
        print(f"✓ Sending analysis request to Gemini for: {laptop_name}")
        
        # Get response from Gemini
        response = self.client_gemini.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        # Parse JSON response
        try:
            # Extract JSON from response (handle potential markdown formatting)
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            analysis = json.loads(response_text.strip())
            print("✓ Successfully parsed Gemini response")
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse Gemini response as JSON: {e}")
            print(f"Raw response: {response.text[:500]}...")
            
            # Return a structured error response
            return {
                "laptop_name": laptop_name,
                "error": "Failed to parse LLM response",
                "raw_response": response.text[:1000]
            }
    
    
    def full_analysis_pipeline(self, laptop_query: str, n_results: int = 5) -> Dict:
        """
        Complete pipeline: Query ChromaDB → Analyze with Gemini.
        
        Args:
            laptop_query: Laptop to analyze (e.g., "Lenovo Legion Y540")
            n_results: Number of ChromaDB results to use as context
            
        Returns:
            Complete analysis dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting Full Analysis Pipeline for: {laptop_query}")
        print(f"{'='*60}\n")
        
        # Step 1: Query ChromaDB
        print("STEP 1: Querying ChromaDB...")
        context_docs = self.query_chromadb(laptop_query, n_results)
        
        if not context_docs:
            print("✗ No relevant documents found in ChromaDB")
            return {
                "laptop_name": laptop_query,
                "error": "No relevant data found in knowledge base"
            }
        
        # Step 2: Analyze with Gemini
        print("\nSTEP 2: Analyzing with Google Gemini...")
        analysis = self.analyze_laptop(laptop_query, context_docs)
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60 + "\n")
        
        return analysis


def save_analysis_to_file(analysis: Dict, output_path: str | None = None):
    """
    Save analysis results to a JSON file.
    
    Args:
        analysis: Analysis dictionary to save
        output_path: Path to save to. If None, generates from laptop_name in analysis.
    """
    if output_path is None:
        laptop_name = analysis.get("laptop_name", "unknown")
        output_path = get_analysis_output_path(laptop_name)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, indent=2, fp=f)
    print(f"Analysis saved to: {output_path}")


def print_analysis_summary(analysis: Dict):
    """Print a formatted summary of the analysis."""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60 + "\n")
    
    if "error" in analysis:
        print(f"⚠ Error: {analysis['error']}")
        return
    
    print(f"📱 Laptop: {analysis.get('laptop_name', 'Unknown')}")
    print(f"📊 Sentiment Score: {analysis.get('sentiment_score', 'N/A')}/100")
    print(f"💭 {analysis.get('sentiment_explanation', 'No explanation')}\n")
    
    print("✅ PROS:")
    for i, pro in enumerate(analysis.get('pros', []), 1):
        print(f"   {i}. {pro}")
    
    print("\n❌ CONS:")
    for i, con in enumerate(analysis.get('cons', []), 1):
        print(f"   {i}. {con}")
    
    print("\n🔑 KEY THEMES:")
    for theme in analysis.get('key_themes', []):
        print(f"   • {theme}")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   {analysis.get('user_recommendation', 'No recommendation available')}")
    
    print("\n" + "="*60 + "\n")


def main():


    """Main execution function."""
    
    try:
        # Initialize analyzer
        analyzer = LaptopAnalyzer(collection_name="laptop_knowledge")
        
        # Run analysis
        laptop_to_analyze = "legion Y540"
        analysis = analyzer.full_analysis_pipeline(
            laptop_query=laptop_to_analyze,
            n_results=3  # Use top 3 most relevant documents
        )
        
        # Display results
        print_analysis_summary(analysis)
        
        # Save to file (auto-generates path based on laptop name)
        save_analysis_to_file(analysis)
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()