import subprocess
import json
import os
import sys

def search_news(query: str) -> dict:
    """
    Call news.py to search for news articles
    """
    try:
        # Get the directory where news_search.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        news_script = os.path.join(current_dir, 'news.py')
        
        # Call news.py as a subprocess
        process = subprocess.run(
            [sys.executable, news_script, query],
            capture_output=True,
            text=True
        )
        
        # Check for errors
        if process.returncode != 0:
            error_msg = process.stderr or "Unknown error"
            return {
                "error": f"news.py failed: {error_msg}",
                "metadata": {
                    "search_used": False,
                    "num_search_results": 0,
                    "execution_time": 0
                }
            }
        
        # Try to parse JSON output
        try:
            return json.loads(process.stdout)
        except json.JSONDecodeError:
            # Return a properly formatted error response
            return {
                "error": "Invalid JSON output from news.py",
                "metadata": {
                    "search_used": False,
                    "num_search_results": 0,
                    "execution_time": 0,
                    "debug_info": {
                        "stdout": process.stdout,
                        "stderr": process.stderr
                    }
                }
            }
            
    except Exception as e:
        return {
            "error": f"Error running news.py: {str(e)}",
            "metadata": {
                "search_used": False,
                "num_search_results": 0,
                "execution_time": 0
            }
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Search for news articles")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    args = parser.parse_args()
    
    result = search_news(args.query)
    # Ensure we always output valid JSON
    print(json.dumps(result))

if __name__ == "__main__":
    main()
