import os
import sys
import json
from typing import List, Dict, Any
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from exa_py import Exa
from utils import *
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Augmented LLM Script with Function Calling

This script implements the architecture shown in the diagram:
- Input (User Query) -> LLM -> Output (Response)
- With LLM connected to external tools (Exa search in this case)
- Uses Function Calling to determine if search is needed

Flow:
1. User provides a query
2. LLM determines if external information is needed (using function calling)
3. If needed, LLM uses Exa search to retrieve relevant information
4. LLM incorporates the retrieved information into its response
5. Final response is returned to the user
"""

# Load environment variables
load_dotenv()

# Exa API configuration
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")

# Validate API key
if not EXA_API_KEY:
    logger.error("EXA_API_KEY is not set")

class ExaSearchTool:
    """Tool for searching the web using Exa API"""
    
    def __init__(self, api_key: str):
        """Initialize the search tool with API key"""
        self.api_key = api_key
        self.client = Exa(api_key=api_key)
        
    def search(self, query: str, num_results: int = 5) -> list[dict[str, any]]:
        """
        Search the web using Exa API
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        # Add date range to query to get recent results
        today = datetime.today()
        three_months_ago = today - timedelta(days=90)
        date_range = f"after:{three_months_ago.strftime('%Y-%m-%d')}"
        full_query = f"{query} {date_range}"
        
        try:
            # Perform search with date range
            results = self.client.search_and_contents(
                query=full_query,
                num_results=num_results,
                use_autoprompt=True
            )
            # If no results, try with a broader search
            if not results.results:
                results = self.client.search_and_contents(
                    query=full_query,
                    num_results=num_results,
                    use_autoprompt=True,
                    highlights=True,
                    text=True,
                    type="news"
                )
            
            # If still no results, try without date range
            if not results.results:
                results = self.client.search_and_contents(
                    query=query,
                    num_results=num_results,
                    use_autoprompt=True,
                    highlights=True,
                    text=True,
                    type="news"
                )
            
            # Format results
            formatted_results = []
            for result in results.results:
                # Parse the published date
                try:
                    published_date = datetime.strptime(result.published_date, "%Y-%m-%dT%H:%M:%S%z")
                    date_str = published_date.strftime("%B %d, %Y")
                except (ValueError, TypeError):
                    date_str = datetime.today().strftime("%B %d, %Y")
                
                # Clean and format the text
                text = result.text or ""
                if result.highlights:
                    text = " ... ".join(result.highlights)
                
                formatted_results.append({
                    "title": result.title,
                    "url": result.url,
                    "date": date_str
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

class AugmentedLLM:
    """LLM augmented with external tool access"""
    
    def __init__(self, search_tool: ExaSearchTool, model: str = "gpt-4o", verbose: bool = False):
        """Initialize the augmented LLM"""
        self.search_tool = search_tool
        self.model = model
        self.conversation_history = []
        self.verbose = verbose
        self.last_search_results = []
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for inclusion in prompt"""
        if not results:
            return "No results found."
        
        formatted_text = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted_text += f"[{i}] {result['title']}\n"
            formatted_text += f"URL: {result['url']}\n"
            formatted_text += f"Date: {result['date']}\n\n"
        
        return formatted_text
    
    def _search_decision_and_query(self, query: str) -> tuple:
        """
        Determine if search should be used and generate optimized search query using parallel function calling
        
        Returns:
            tuple: (should_search, optimized_query)
        """
        # Define the tool functions
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "perform_search",
                    "description": "Determine if web search is needed to answer the user's query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "should_search": {
                                "type": "boolean",
                                "description": "Whether external search is needed to provide an accurate and up-to-date answer."
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of why search is or isn't needed."
                            }
                        },
                        "required": ["should_search", "reasoning"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "optimize_search_query",
                    "description": "Generate an optimized search query to find the most relevant and recent (~3 months) information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "optimized_query": {
                                "type": "string",
                                "description": "An optimized version of the user's query crafted for web search."
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Brief explanation of how the query was optimized."
                            }
                        },
                        "required": ["optimized_query", "explanation"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]
        
        # Prepare the system message
        system_message = """
        You are an AI assistant that performs two functions:
        
        1. DETERMINE if external search is needed based on these criteria:
           - The query asks about current events or recent information
           - The query requests factual information that might change over time
           - The query is about a specific subject you might have limited information about
           
           SEARCH IS NOT NEEDED if:
           - The query is about general knowledge that doesn't change
           - The query is asking for logical reasoning, opinions, or creative content
           - The query is about well-established concepts or definitions
        
        2. OPTIMIZE the search query by:
           - Focusing on the key information needs
           - Removing unnecessary words and phrases
           - Using specific terms that will yield better search results
           - Adding relevant keywords if the original query is ambiguous
           - Ensuring the query is recent (within the last 3 months of """ + datetime.today().strftime('%Y-%m-%d') + """)
        
        Perform BOTH functions for every user query.
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
        
        # Call the model to determine if search is needed and to optimize the query
        # Note: Setting parallel_tool_calls=True to allow multiple function calls in parallel
        response = get_chat_completion(
            messages,
            model=self.model,
            tools=tools,
            tool_choice="auto",  # Allow the model to decide which tools to use
            parallel_tool_calls=True  # Enable parallel function calling
        )
        
        # Default values
        should_search = False
        optimized_query = query  # Default to original query
        search_reasoning = ""
        query_explanation = ""
        
        # Parse the tool calls from the response
        tool_calls = getattr(response, 'tool_calls', None)
        if tool_calls and len(tool_calls) > 0:
            for tool_call in tool_calls:
                try:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "perform_search":
                        should_search = function_args.get('should_search', False)
                        search_reasoning = function_args.get('reasoning', "No reasoning provided")
                        
                    elif function_name == "optimize_search_query":
                        optimized_query = function_args.get('optimized_query', query)
                        query_explanation = function_args.get('explanation', "No explanation provided")
                        
                except (json.JSONDecodeError, AttributeError, KeyError) as e:
                    logger.error(f"Error parsing function call: {e}")
            
            if self.verbose:
                logger.info(f"Search decision: {should_search}, Reasoning: {search_reasoning}")
                logger.info(f"Optimized query: '{optimized_query}', Explanation: {query_explanation}")
        else:
            # If no tool call was made, fallback to assuming no search is needed
            logger.info("No tool calls were made by the model")
        
        return should_search, optimized_query
    
    def _should_use_search(self, query: str) -> bool:
        """Determine if search should be used for this query (legacy method)"""
        should_search, _ = self._search_decision_and_query(query)
        return should_search
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to the user query, using search when appropriate
        
        Args:
            query: User query
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        search_used = False
        search_results = []
        optimized_query = query  # Default to original query
        
        # Add user query to conversation
        self.conversation_history.append({"role": "user", "content": query})
        
        # Determine if search should be used and get optimized query
        should_search, optimized_query = self._search_decision_and_query(query)
        if should_search:
            search_used = True
            search_results = self.search_tool.search(optimized_query)
            search_text = self._format_search_results(search_results)
            
            
            # Prepare system message with search results
            system_message = f"""
            You are an AI assistant augmented with the ability to search the web for information.
            For the user's query, you have the following search results that may contain relevant information.
            Use these results to inform your response, and cite sources when appropriate.
            
            {search_text}
            
            IMPORTANT GUIDELINES:
            1. Provide DIRECT and SPECIFIC answers based on the search results - do not just tell the user where they can find information.
            2. Extract precise facts, dates, numbers, and details from the search results.
            3. If a search result contains the exact answer (like a date, time, location, or fact), state it explicitly.
            4. Do NOT respond with phrases like "To find out..." or "You can check..."
            5. If the search results don't contain relevant information, use your own knowledge but be direct.
            6. Cite sources by referencing the search result number when providing information.
            7. Do NOT include citation numbers like [1], [2], etc. and DO NOT include the sources used anywhere in the summary. Just include the summary.
            
            Example of a BAD response: "According to [1], the Lakers play next on March 18."
            Example of a GOOD response: "The Lakers play next on March 18 at 7:30 PM against the Phoenix Suns at Crypto.com Arena."
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
        else:
            # Standard prompt without external search results
            if self.verbose:
                logger.info("Answering without external search")
            system_message = """You are a helpful AI assistant. Provide a direct and specific answer to the user's query based on your knowledge.
            
            IMPORTANT GUIDELINES:
            1. Be direct and specific in your answers.
            2. Provide precise facts, dates, numbers, and details when available.
            3. Do NOT respond with phrases like 'To find out...' or 'You can check...'
            4. If you don't know the exact answer, say so clearly but still provide your best response.
            
            Example of a BAD response: "To find out about quantum physics, you can read books on the subject."
            Example of a GOOD response: "Quantum physics is the study of matter and energy at the most fundamental level. It describes how particles like electrons and photons behave in ways that differ from classical physics."""
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
        
        # Generate response
        response = get_chat_completion(messages, model=self.model)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        execution_time = time.time() - start_time
        
        return {
            "query": query,
            "response": response,
            "search_used": search_used,
            "num_search_results": len(search_results) if search_used else 0,
            "execution_time": execution_time,
            "search_results": search_results,  # Always include search_results
            "metadata": {
                "search_used": search_used,
                "num_search_results": len(search_results) if search_used else 0,
                "execution_time": execution_time,
                "date_range": {
                    "from": "January 20, 2025",
                    "to": datetime.today().strftime('%B %d, %Y')
                }
            }
        }

# Create FastAPI app
app = FastAPI(title="News LLM API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Server is running"}

@app.get("/")
async def root():
    return JSONResponse({
        "status": "ok",
        "message": "News LLM API is running",
        "endpoints": {
            "/": "This help message",
            "/health": "Health check endpoint",
            "/api/search": "POST - Search for news articles"
        }
    })

class SearchQuery(BaseModel):
    query: str

@app.post("/api/search")
async def search_news(query: SearchQuery):
    """Search for news articles"""
    try:
        logger.info(f"Received search query: {query.query}")
        
        # Log API key status (without revealing the keys)
        logger.info(f"EXA_API_KEY present: {bool(EXA_API_KEY)}")
        
        if not EXA_API_KEY:
            raise ValueError("Missing required API keys")

        # Initialize tools
        logger.info("Initializing search tool...")
        search_tool = ExaSearchTool(api_key=EXA_API_KEY)
        
        logger.info("Initializing LLM...")
        augmented_llm = AugmentedLLM(search_tool=search_tool, verbose=True)
        
        # Generate response
        logger.info("Generating response...")
        result = augmented_llm.generate_response(query.query)
        logger.info("Response generated successfully")
        
        # Format articles
        articles = []
        if result.get("search_used") and result.get("num_search_results", 0) > 0:
            logger.info(f"Processing {result.get('num_search_results', 0)} search results")
            for search_result in result.get("search_results", []):
                try:
                    articles.append({
                        "title": search_result.get("title", ""),
                        "url": search_result.get("url", ""),
                        "date": search_result.get("date", "")
                    })
                except Exception as article_error:
                    logger.error(f"Error formatting article: {article_error}")
                    continue
        
        response_data = {
            "results": articles,
            "response": result.get("response", ""),
            "metadata": {
                "search_used": result.get("search_used", False),
                "num_search_results": result.get("num_search_results", 0),
                "execution_time": result.get("execution_time", 0),
                "date_range": result.get("date_range", {"from": "", "to": ""})
            }
        }
        logger.info("Successfully prepared response")
        return response_data
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error in search_news: {str(e)}\nTraceback:\n{error_trace}")
        if "api_key" in str(e).lower():
            raise HTTPException(status_code=500, detail="API key configuration error")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

def main():
    """Main function to run the augmented LLM"""
    try:
        # Initialize tools
        logger.info("Initializing search tool...")
        search_tool = ExaSearchTool(api_key=EXA_API_KEY)
        
        logger.info("Initializing LLM...")
        augmented_llm = AugmentedLLM(search_tool=search_tool, verbose=True)
        
        # Check if query is provided as command-line argument
        if len(sys.argv) > 1:
            query = sys.argv[1]
            result = augmented_llm.generate_response(query)
            
            # Format the response into articles
            articles = []
            if result["search_used"] and result["num_search_results"] > 0:
                # Format each search result as an article
                for search_result in result.get("search_results", []):
                    article = f"{search_result['title']} | {search_result['date']} | {search_result['url']} | {search_result['text']}"
                    articles.append(article)
            else:
                # Use the response as a single article
                articles.append(f"AI Response | {datetime.today().strftime('%B %d, %Y')} | # | {result['response']}")
            
            # Always output valid JSON
            json_response = {
                "results": articles,
                "response": result["response"],
                "metadata": {
                    "search_used": result["search_used"],
                    "num_search_results": result["num_search_results"],
                    "execution_time": result["execution_time"],
                    "date_range": result["metadata"]["date_range"]
                }
            }
            sys.stdout.write(json.dumps(json_response))
            return
        
        # Run interactive mode
        while True:
            # Get user query
            query = input("User: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            # Generate response
            result = augmented_llm.generate_response(query)
            
            # Return result in JSON format with no extra text
            json_response = {
                "results": [result["response"]],
                "metadata": {
                    "search_used": result["search_used"],
                    "num_search_results": result["num_search_results"],
                    "execution_time": result["execution_time"],
                    "date_range": result["metadata"]["date_range"]
                }
            }
            sys.stdout.write(json.dumps(json_response))
            return
    except Exception as e:
        # Return error in JSON format
        error_response = {
            "error": str(e),
            "metadata": {
                "search_used": False,
                "num_search_results": 0,
                "execution_time": 0,
                "date_range": {
                    "from": "January 20, 2025",
                    "to": datetime.today().strftime('%B %d, %Y')
                }
            }
        }
        sys.stdout.write(json.dumps(error_response))
        
if __name__ == "__main__":
    main()
