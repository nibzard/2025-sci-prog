Guardian News Scraper and Analyzer (Marimo App)

This Marimo application scrapes news headlines from The Guardian's Europe section, extracts relevant article information, and then uses the Google Gemini API to analyze the scraped content, providing summaries and identifying prevalent themes.
Features

    Web Scraping: Utilizes the Steel API to scrape HTML content from a specified URL (The Guardian's Europe page).
    Data Extraction: Parses the HTML using BeautifulSoup to identify and extract news article titles and URLs.
    AI Analysis: Integrates with the Google Gemini API to analyze the extracted news articles, generating summaries and identifying overarching themes.
    Interactive Interface: Built as a Marimo application for easy execution and display of results.

Setup

To run this application, you will need the following:

    Python Environment: Ensure you have Python 3.8+ installed.

    Dependencies: Install the required Python packages:

    pip install marimo requests python-dotenv beautifulsoup4 google-generativeai

    API Keys: This application requires two API keys:
        Steel API Key: For web scraping. Obtain one from Steel API.
        Google Gemini API Key: For AI content analysis. Obtain one from Google AI Studio.

    Environment Variables: Create a .env file in the same directory as this Marimo application, and add your API keys to it:

    steel_api="YOUR_STEEL_API_KEY"
    google_api="YOUR_GEMINI_API_KEY"

    Note: In Colab, you might need to manually set these as environment variables or use Colab's 'Secrets' feature.

How to Run

    Save the Application: Save the Marimo application code as a .py file (e.g., guardian_news_app.py).
    Run with Marimo: Open your terminal or command prompt, navigate to the directory where you saved the file, and run the application using Marimo:

    marimo run guardian_news_app.py

    Access the App: Marimo will typically open a new tab in your web browser displaying the interactive application. You can then run the cells within the Marimo interface.

Code Structure

The application is organized into several Marimo cells:

    Imports: Imports all necessary libraries.
    API Key and URL Configuration: Loads API keys from environment variables and defines the target URL for scraping.
    Web Scraping: Uses the Steel API to fetch HTML content from The Guardian.
    HTML Parsing: Uses BeautifulSoup to extract article titles and links from the scraped HTML.
    Gemini Analysis: Sends the extracted news items to the Google Gemini API for summarization and theme identification.
