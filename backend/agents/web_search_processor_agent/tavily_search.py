import requests
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List, Dict, Any

class TavilySearchAgent:
    """
    Enhanced Tavily search agent for medical information retrieval with improved source formatting.
    """
    def __init__(self, config=None):
        """
        Initialize the Tavily search agent.

        Args:
            config: Configuration object containing web search settings
        """
        self.config = config
        self.max_results = getattr(config, 'max_results', 5) if config else 5
        self.content_max_length = getattr(config, 'content_max_length', 500) if config else 500

    def _format_search_results(self, search_docs: List[Dict[str, Any]]) -> str:
        """
        Format search results in a structured way for better LLM processing.

        Args:
            search_docs: List of search result dictionaries

        Returns:
            Formatted string with structured search results
        """
        if not search_docs:
            return "No relevant results found."

        formatted_results = []
        for i, result in enumerate(search_docs, 1):
            title = result.get("title", "Unknown Title")
            url = result.get("url", "")
            content = result.get("content", "")
            score = result.get("score", 0.0)

            # Clean and truncate content if too long
            content = content.strip()
            if len(content) > self.content_max_length:
                content = content[:self.content_max_length] + "..."

            formatted_result = f"""
RESULT {i}:
Title: {title}
URL: {url}
Relevance Score: {score:.3f}
Content: {content}
---"""
            formatted_results.append(formatted_result)

        return "\n".join(formatted_results)

    def search_tavily(self, query: str) -> str:
        """
        Perform a general web search using Tavily API with enhanced formatting.

        Args:
            query: Search query string

        Returns:
            Formatted search results string
        """
        tavily_search = TavilySearchResults(max_results=self.max_results)

        try:
            # Strip any surrounding quotes from the query
            query = query.strip('"\'')
            # print("Printing query:", query)
            search_docs = tavily_search.invoke(query)

            if len(search_docs):
                return self._format_search_results(search_docs)
            return "No relevant results found."

        except Exception as e:
            return f"Error retrieving web search results: {e}"