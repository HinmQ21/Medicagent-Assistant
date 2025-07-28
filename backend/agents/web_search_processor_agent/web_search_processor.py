import os
from .web_search_agent import WebSearchAgent
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class WebSearchProcessor:
    """
    Processes web search results and routes them to the appropriate LLM for response generation.
    """
    
    def __init__(self, config):
        self.web_search_agent = WebSearchAgent(config)
        
        # Initialize LLM for processing web search results
        self.llm = config.web_search.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Build the prompt for the web search.
        
        Args:
            query: User query
            chat_history: chat history
            
        Returns:
            Complete prompt string
        """
        # Add chat history if provided
        # print("Chat History:", chat_history)
            
        # Build the prompt
        prompt = f"""Here are the last few messages from our conversation:

        {chat_history}

        The user asked the following question:

        {query}

        Summarize them into a single, well-formed question only if the past conversation seems relevant to the current query so that it can be used for a web search.
        Keep it concise and ensure it captures the key intent behind the discussion.
        """

        return prompt
    
    def _build_enhanced_response_prompt(self, query: str, web_results: str) -> str:
        """
        Build an enhanced prompt for generating clean response with proper source citations.

        Args:
            query: User query
            web_results: Raw web search results

        Returns:
            Enhanced prompt string
        """
        prompt = f"""You are an AI medical assistant specialized in providing accurate, evidence-based medical information.

Your task is to analyze the web search results and provide a comprehensive, well-structured response that includes proper source citations.

CRITICAL INSTRUCTIONS:
1. Analyze the provided web search results carefully
2. Synthesize the information into a coherent, helpful response
3. Ensure medical accuracy and reliability - only use information from credible sources
4. ALWAYS include source citations for every major claim or piece of information
5. Prioritize information from reputable medical sources (medical journals, health organizations, hospitals, government health agencies, etc.)
6. If conflicting information exists, acknowledge it and explain the differences
7. Use clear, accessible language while maintaining medical accuracy
8. Structure your response with clear sections if the topic is complex
9. Be transparent about limitations and recommend consulting healthcare professionals when appropriate

MANDATORY RESPONSE FORMAT:
1. **Direct Answer**: Start with a clear, direct answer to the user's question
2. **Detailed Explanation**: Provide comprehensive information with supporting evidence
3. **Important Notes**: Include relevant context, warnings, or additional considerations
4. **References**: End with a "##### Sources:" section listing all referenced sources

CITATION FORMAT:
- Use inline citations like [1], [2], etc. throughout your response
- At the end, list sources as:
##### Sources:
[1] Title - URL
[2] Title - URL
etc.

QUALITY STANDARDS:
- Only cite information that directly appears in the search results
- Distinguish between different types of sources (research studies, health organizations, news articles)
- If information is limited or unclear, state this explicitly
- Always recommend consulting healthcare professionals for personalized medical advice

CRITICAL: Respond ONLY with the medical information in the exact format specified above. Do NOT include:
- Any meta-commentary like "The response is appropriate"
- Query repetition like "ORIGINAL USER QUERY:" or "Tell me about..."
- Response prefixes like "CHATBOT RESPONSE:", "Here is the response:", "Here's the information:"
- Any evaluation or commentary text
- Any introductory phrases

Start your response IMMEDIATELY with "**Direct Answer:**" followed by the actual medical content.

USER QUERY: {query}

WEB SEARCH RESULTS:
{web_results}

Please provide a comprehensive response following the mandatory format above:"""

        return prompt

    def _post_process_response(self, response_content: str) -> str:
        """
        Post-process the LLM response to ensure proper formatting and remove unnecessary text.

        Args:
            response_content: Raw response from LLM

        Returns:
            Processed response with improved formatting
        """
        import re

        # Remove common unwanted prefixes/suffixes
        unwanted_phrases = [
            "The response is appropriate.",
            "ORIGINAL USER QUERY:",
            "CHATBOT RESPONSE:",
            "Here is the response:",
            "Response:",
            "The answer is:",
            "Based on the search results:",
            "Here's the information:",
            "According to the search results:",
            "The search results show:",
            "Here is a comprehensive response:",
            "Here's a comprehensive response:",
            "Please find the response below:",
            "The following is the response:",
        ]

        # Clean the response
        cleaned_response = response_content.strip()

        # Remove unwanted phrases from beginning (case insensitive)
        for phrase in unwanted_phrases:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            if pattern.match(cleaned_response):
                cleaned_response = pattern.sub("", cleaned_response, count=1).strip()

        # Remove unwanted phrases from anywhere in text (case insensitive)
        for phrase in unwanted_phrases:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            cleaned_response = pattern.sub("", cleaned_response).strip()

        # Remove any lines that contain only meta-text patterns
        lines = cleaned_response.split('\n')
        filtered_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Skip lines that are purely meta-commentary
            if (line_stripped.startswith("ORIGINAL USER QUERY:") or
                line_stripped.startswith("CHATBOT RESPONSE:") or
                line_stripped == "The response is appropriate." or
                line_stripped.startswith("Here is") or
                line_stripped.startswith("Here's") or
                line_stripped.startswith("Please find") or
                line_stripped.startswith("The following is")):
                continue
            filtered_lines.append(line)

        cleaned_response = '\n'.join(filtered_lines).strip()

        # Remove multiple consecutive newlines
        cleaned_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_response)

        # Remove any remaining empty lines at the beginning
        cleaned_response = cleaned_response.lstrip('\n').strip()

        # Ensure there's a sources section if it's missing
        if "##### Sources:" not in cleaned_response and "Sources:" not in cleaned_response:
            # Try to extract URLs from the response and add a basic sources section
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', cleaned_response)
            if urls:
                cleaned_response += "\n\n##### Sources:\n"
                for i, url in enumerate(set(urls), 1):
                    cleaned_response += f"[{i}] {url}\n"

        # Add disclaimer if not present
        disclaimer = "\n\n**Note**: This information is for educational purposes only and cannot replace professional medical advice. Please consult with a healthcare professional for appropriate medical guidance."

        if "note:" not in cleaned_response.lower() and "consult" not in cleaned_response.lower() and "disclaimer" not in cleaned_response.lower():
            cleaned_response += disclaimer

        return cleaned_response

    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Fetches web search results, processes them using LLM, and returns a clean user-friendly response with citations.
        """
        # print(f"[WebSearchProcessor] Fetching web search results for: {query}")
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)
        # print("Web Search Query Prompt:", web_search_query_prompt)
        web_search_query = self.llm.invoke(web_search_query_prompt)
        # print("Web Search Query:", web_search_query)

        # Retrieve web search results
        web_results = self.web_search_agent.search(web_search_query.content)

        # print(f"[WebSearchProcessor] Fetched results: {web_results}")

        # Construct enhanced prompt for better response generation
        llm_prompt = self._build_enhanced_response_prompt(query, web_results)

        # Invoke the LLM to process the results
        response = self.llm.invoke(llm_prompt)

        # Post-process the response to ensure proper formatting and remove unwanted text
        processed_response = self._post_process_response(response.content)

        # Create a response object with the processed content
        response.content = processed_response

        return response
