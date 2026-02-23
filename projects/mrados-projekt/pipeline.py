# pipeline.py
"""
Central Pipeline Controller for Reddit -> RAG -> LLM Analysis.

This module orchestrates the full pipeline:
1. Reddit Search Scraper - Search for posts about a laptop
2. URL Deduplication - Extract unique URLs
3. Reddit Post Scraper - Scrape each post once
4. Markdown Generator - Convert posts to markdown
5. ChromaDB Storage - Embed and store markdown
6. LLM Analyzer - Analyze sentiment with Gemini
"""

import os
import json
import time
from typing import Set

from utils import (
    laptop_name_to_slug,
    get_search_results_path,
    get_scraped_urls_path,
    get_laptop_markdown_dir,
    get_analysis_output_path
)
from scrape_query_reddit import scrape_reddit_queries, extract_unique_urls
from reddit_post_scrapper import scrape_reddit_post
from md_generator import save_post_as_markdown


class LaptopAnalysisPipeline:
    """
    Orchestrates the complete Reddit -> RAG -> LLM analysis pipeline.
    
    Features:
    - Skip scraping if search results already exist
    - Track and skip already-scraped URLs
    - Organize markdown files by laptop
    - Full pipeline with optional step skipping
    """
    
    def __init__(self, laptop_name: str, data_dir: str = "data", knowledge_dir: str = "knowledge/reddit"):
        """
        Initialize the pipeline for a specific laptop.
        
        Args:
            laptop_name: Name of the laptop (e.g., "Lenovo Legion Y540")
            data_dir: Directory for storing JSON data files
            knowledge_dir: Base directory for markdown files
        """
        self.laptop_name = laptop_name
        self.laptop_slug = laptop_name_to_slug(laptop_name)
        self.data_dir = data_dir
        self.knowledge_dir = knowledge_dir
        
        # File paths
        self.search_results_path = get_search_results_path(laptop_name, data_dir)
        self.scraped_urls_path = get_scraped_urls_path(laptop_name, data_dir)
        self.markdown_dir = get_laptop_markdown_dir(laptop_name, knowledge_dir)
        self.analysis_output_path = get_analysis_output_path(laptop_name)
        
        # Track state
        self.search_results = []
        self.all_urls = []
        self.scraped_urls: Set[str] = set()
        self.scraped_posts = []
        self.markdown_files = []
        
        # Load existing scraped URLs
        self._load_scraped_urls()
    
    def _load_scraped_urls(self) -> None:
        """Load previously scraped URLs from disk."""
        if os.path.exists(self.scraped_urls_path):
            try:
                with open(self.scraped_urls_path, "r", encoding="utf-8") as f:
                    self.scraped_urls = set(json.load(f))
                print(f"Loaded {len(self.scraped_urls)} previously scraped URLs")
            except Exception as e:
                print(f"Error loading scraped URLs: {e}")
                self.scraped_urls = set()
    
    def _save_scraped_urls(self) -> None:
        """Save scraped URLs to disk for future reference."""
        os.makedirs(self.data_dir, exist_ok=True)
        try:
            with open(self.scraped_urls_path, "w", encoding="utf-8") as f:
                json.dump(list(self.scraped_urls), f, indent=2)
            print(f"Saved {len(self.scraped_urls)} scraped URLs to {self.scraped_urls_path}")
        except Exception as e:
            print(f"Error saving scraped URLs: {e}")
    
    # =========================================================================
    # STEP 1: Reddit Search
    # =========================================================================
    def step1_search_reddit(self, force_refresh: bool = False) -> list[dict]:
        """
        Step 1: Search Reddit for posts about this laptop.
        
        Args:
            force_refresh: If True, scrape even if cached data exists
            
        Returns:
            Search results data
        """
        print(f"\n{'='*60}")
        print(f"STEP 1: Reddit Search for '{self.laptop_name}'")
        print(f"{'='*60}")
        
        self.search_results = scrape_reddit_queries(
            self.laptop_name, 
            force_refresh=force_refresh,
            data_dir=self.data_dir
        )
        
        return self.search_results
    
    # =========================================================================
    # STEP 2: URL Deduplication
    # =========================================================================
    def step2_extract_urls(self) -> list[str]:
        """
        Step 2: Extract unique URLs from search results.
        
        Returns:
            List of unique Reddit post URLs
        """
        print(f"\n{'='*60}")
        print(f"STEP 2: URL Extraction & Deduplication")
        print(f"{'='*60}")
        
        if not self.search_results:
            print("No search results loaded. Run step1_search_reddit first.")
            return []
        
        self.all_urls = extract_unique_urls(self.search_results)
        
        # Filter out already-scraped URLs
        new_urls = [url for url in self.all_urls if url not in self.scraped_urls]
        
        print(f"Total unique URLs: {len(self.all_urls)}")
        print(f"Already scraped: {len(self.all_urls) - len(new_urls)}")
        print(f"New URLs to scrape: {len(new_urls)}")
        
        return new_urls
    
    # =========================================================================
    # STEP 3: Post Scraping
    # =========================================================================
    def step3_scrape_posts(self, urls: list[str] | None = None, delay: float = 2.0) -> list[dict]:
        """
        Step 3: Scrape each Reddit post.
        
        Args:
            urls: List of URLs to scrape (if None, uses new URLs from step2)
            delay: Delay between requests in seconds
            
        Returns:
            List of scraped post dictionaries
        """
        print(f"\n{'='*60}")
        print(f"STEP 3: Scraping Reddit Posts")
        print(f"{'='*60}")
        
        if urls is None:
            urls = [url for url in self.all_urls if url not in self.scraped_urls]
        
        if not urls:
            print("No new URLs to scrape.")
            return []
        
        print(f"Scraping {len(urls)} posts...")
        
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Scraping: {url[:60]}...")
            
            try:
                post = scrape_reddit_post(url)
                
                if post:
                    self.scraped_posts.append(post)
                    self.scraped_urls.add(url)
                    print(f"  -> Success: {post['title'][:50]}...")
                else:
                    print(f"  -> Failed to scrape (no content)")
                    
            except Exception as e:
                print(f"  -> Error: {e}")
            
            # Rate limiting
            if i < len(urls):
                time.sleep(delay)
        
        # Save updated scraped URLs
        self._save_scraped_urls()
        
        print(f"\nSuccessfully scraped {len(self.scraped_posts)} posts")
        return self.scraped_posts
    
    # =========================================================================
    # STEP 4: Markdown Generation
    # =========================================================================
    def step4_generate_markdown(self, posts: list[dict] | None = None) -> list[str]:
        """
        Step 4: Convert scraped posts to markdown files.
        
        Args:
            posts: List of post dictionaries (if None, uses scraped_posts)
            
        Returns:
            List of created markdown file paths
        """
        print(f"\n{'='*60}")
        print(f"STEP 4: Generating Markdown Files")
        print(f"{'='*60}")
        
        if posts is None:
            posts = self.scraped_posts
        
        if not posts:
            print("No posts to convert to markdown.")
            return []
        
        print(f"Converting {len(posts)} posts to markdown...")
        print(f"Output directory: {self.markdown_dir}")
        
        for post in posts:
            try:
                filepath = save_post_as_markdown(
                    post,
                    laptop_name=self.laptop_name,
                    base_dir=self.knowledge_dir
                )
                self.markdown_files.append(filepath)
            except Exception as e:
                print(f"Error creating markdown for '{post.get('title', 'unknown')}': {e}")
        
        print(f"\nCreated {len(self.markdown_files)} markdown files")
        return self.markdown_files
    
    # =========================================================================
    # STEP 5: ChromaDB Storage
    # =========================================================================
    def step5_store_in_chromadb(self, md_files: list[str] | None = None, collection_name: str = "laptop_knowledge") -> int:
        """
        Step 5: Store markdown files in ChromaDB.
        
        Args:
            md_files: List of markdown file paths (if None, uses markdown_files)
            collection_name: ChromaDB collection name
            
        Returns:
            Number of documents stored
        """
        print(f"\n{'='*60}")
        print(f"STEP 5: Storing in ChromaDB")
        print(f"{'='*60}")
        
        # Import here to avoid circular imports and allow graceful failure
        try:
            from vectorstore.chromadb_store import store_markdown_in_chroma
        except ImportError as e:
            print(f"Error importing chromadb_store: {e}")
            print("Make sure chromadb and sentence-transformers are installed.")
            return 0
        
        if md_files is None:
            md_files = self.markdown_files
        
        # Also get any existing markdown files for this laptop
        existing_files = []
        if os.path.exists(self.markdown_dir):
            for f in os.listdir(self.markdown_dir):
                if f.endswith('.md'):
                    full_path = os.path.join(self.markdown_dir, f)
                    if full_path not in md_files:
                        existing_files.append(full_path)
        
        all_files = list(set(md_files + existing_files))
        
        if not all_files:
            print("No markdown files to store.")
            return 0
        
        print(f"Storing {len(all_files)} markdown files in ChromaDB...")
        
        stored_count = 0
        for filepath in all_files:
            try:
                # Use absolute path - chromadb_store handles both absolute and relative
                abs_path = os.path.abspath(filepath)
                if store_markdown_in_chroma(abs_path, collection_name, self.laptop_name):
                    stored_count += 1
            except Exception as e:
                print(f"Error storing {filepath}: {e}")
        
        print(f"\nStored {stored_count} documents in ChromaDB")
        return stored_count
    
    # =========================================================================
    # STEP 6: LLM Analysis
    # =========================================================================
    def step6_analyze_with_llm(self, n_results: int = 5) -> dict:
        """
        Step 6: Analyze laptop sentiment using LLM.
        
        Args:
            n_results: Number of ChromaDB results to use as context
            
        Returns:
            Analysis dictionary with pros, cons, sentiment score
        """
        print(f"\n{'='*60}")
        print(f"STEP 6: LLM Analysis")
        print(f"{'='*60}")
        
        # Import here to avoid circular imports
        try:
            from analysis.llm_analyzer import (
                LaptopAnalyzer, 
                save_analysis_to_file, 
                print_analysis_summary
            )
        except ImportError as e:
            print(f"Error importing llm_analyzer: {e}")
            print("Make sure google-genai and other dependencies are installed.")
            return {"error": str(e)}
        
        try:
            analyzer = LaptopAnalyzer(collection_name="laptop_knowledge")
            analysis = analyzer.full_analysis_pipeline(
                laptop_query=self.laptop_name,
                n_results=n_results
            )
            
            # Save analysis
            save_analysis_to_file(analysis, self.analysis_output_path)
            
            # Print summary
            print_analysis_summary(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"Error during LLM analysis: {e}")
            return {"error": str(e)}
    
    # =========================================================================
    # FULL PIPELINE
    # =========================================================================
    def run_full_pipeline(
        self, 
        force_refresh_search: bool = False,
        scrape_delay: float = 2.0,
        skip_chromadb: bool = False,
        skip_llm: bool = False
    ) -> dict:
        """
        Run the complete pipeline from search to analysis.
        
        Args:
            force_refresh_search: Force re-scraping of search results
            scrape_delay: Delay between post scraping requests
            skip_chromadb: Skip ChromaDB storage step
            skip_llm: Skip LLM analysis step
            
        Returns:
            Summary dictionary with results from each step
        """
        print(f"\n{'#'*60}")
        print(f"# LAPTOP ANALYSIS PIPELINE")
        print(f"# Target: {self.laptop_name}")
        print(f"{'#'*60}")
        
        results = {
            "laptop_name": self.laptop_name,
            "laptop_slug": self.laptop_slug,
            "steps_completed": []
        }
        
        # Step 1: Search Reddit
        self.step1_search_reddit(force_refresh=force_refresh_search)
        results["search_results_count"] = sum(
            len(block.get("results", [])) for block in self.search_results
        )
        results["steps_completed"].append("search")
        
        # Step 2: Extract URLs
        new_urls = self.step2_extract_urls()
        results["unique_urls"] = len(self.all_urls)
        results["new_urls"] = len(new_urls)
        results["steps_completed"].append("deduplicate")
        
        # Step 3: Scrape Posts
        if new_urls:
            self.step3_scrape_posts(new_urls, delay=scrape_delay)
        results["posts_scraped"] = len(self.scraped_posts)
        results["steps_completed"].append("scrape_posts")
        
        # Step 4: Generate Markdown
        if self.scraped_posts:
            self.step4_generate_markdown()
        results["markdown_files"] = len(self.markdown_files)
        results["steps_completed"].append("markdown")
        
        # Step 5: ChromaDB Storage
        if not skip_chromadb:
            stored = self.step5_store_in_chromadb()
            results["chromadb_documents"] = stored
            results["steps_completed"].append("chromadb")
        
        # Step 6: LLM Analysis
        if not skip_llm:
            analysis = self.step6_analyze_with_llm()
            results["analysis"] = analysis
            results["steps_completed"].append("llm_analysis")
        
        # Summary
        print(f"\n{'#'*60}")
        print(f"# PIPELINE COMPLETE")
        print(f"{'#'*60}")
        print(f"Laptop: {self.laptop_name}")
        print(f"Search results: {results.get('search_results_count', 0)}")
        print(f"Unique URLs: {results.get('unique_urls', 0)}")
        print(f"New URLs scraped: {results.get('new_urls', 0)}")
        print(f"Markdown files: {results.get('markdown_files', 0)}")
        print(f"Steps completed: {', '.join(results['steps_completed'])}")
        
        return results


def run_pipeline(laptop_name: str, **kwargs) -> dict:
    """
    Convenience function to run the full pipeline.
    
    Args:
        laptop_name: Name of the laptop to analyze
        **kwargs: Additional arguments passed to run_full_pipeline
        
    Returns:
        Pipeline results dictionary
    """
    pipeline = LaptopAnalysisPipeline(laptop_name)
    return pipeline.run_full_pipeline(**kwargs)


def main():
    """Example usage of the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run laptop analysis pipeline")
    parser.add_argument("laptop", nargs="?", default="Lenovo Legion Y540", 
                        help="Laptop name to analyze")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-scraping of search results")
    parser.add_argument("--skip-chromadb", action="store_true",
                        help="Skip ChromaDB storage")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM analysis")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between scraping requests")
    
    args = parser.parse_args()
    
    results = run_pipeline(
        args.laptop,
        force_refresh_search=args.force_refresh,
        scrape_delay=args.delay,
        skip_chromadb=args.skip_chromadb,
        skip_llm=args.skip_llm
    )
    
    return results


if __name__ == "__main__":
    main()
