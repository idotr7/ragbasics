"""
use_prompt_cache.py - Example usage of prompt caching with different LLM providers

This script demonstrates how to use prompt caching with OpenAI, Google, and Anthropic
when starting a conversation, using the Hebrew PDFs from the Ono data directory.
"""

import os
import time
import argparse
from dotenv import load_dotenv

# Import our LLM implementations
from LLM import (
    OpenAIWithCache, 
    GeminiWithCache, 
    ClaudeWithCache, 
    generate_prompts_from_data,
    preload_ono_data_to_cache
)

# Load environment variables
load_dotenv()

def main():
    """Main function to demonstrate prompt caching."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demonstrate prompt caching with different LLM providers")
    parser.add_argument("--provider", type=str, choices=["openai", "google", "anthropic"], 
                        default="google", help="Which LLM provider to use")
    parser.add_argument("--max-tokens", type=int, default=10000, 
                        help="Maximum token count to include from the PDFs")
    parser.add_argument("--preload", action="store_true", 
                        help="Whether to preload the cache on startup")
    parser.add_argument("--query", type=str, 
                        default="מה הן הפעילויות העיקריות במרכז היזמות?", 
                        help="Query to test (default is in Hebrew)")
    args = parser.parse_args()
    
    # Print startup information
    print(f"Using {args.provider.upper()} as the LLM provider")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Preload cache: {args.preload}")
    print(f"Test query: {args.query}")
    print("-" * 50)
    
    # Initialize the appropriate LLM with caching
    if args.provider == "openai":
        print("Initializing OpenAI with cache...")
        llm = OpenAIWithCache(model="gpt-4o-mini")
    elif args.provider == "google":
        print("Initializing Google Gemini with cache...")
        llm = GeminiWithCache(model="gemini-2.0-flash")
    elif args.provider == "anthropic":
        print("Initializing Anthropic Claude with cache...")
        llm = ClaudeWithCache(model="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unknown provider: {args.provider}")
    
    # Preload the cache if requested
    if args.preload:
        print("Preloading cache with Ono data...")
        # Option 1: Direct approach - preload each provider individually
        prompts = generate_prompts_from_data("Ono data", args.max_tokens)
        llm.preload_cache(prompts)
        
        # Option 2: Alternative approach using the helper function
        # preload_ono_data_to_cache(args.provider, args.max_tokens)
    
    # Test the model with a query (first time)
    print("\nFirst query (should be slower if cache isn't preloaded):")
    start_time = time.time()
    response = llm.get_completion(args.query)
    first_time = time.time() - start_time
    print(f"Response time: {first_time:.4f} seconds")
    print(f"Response: {response[:150]}...")
    
    # Test the model with the same query again (second time, should be faster due to cache)
    print("\nSecond query with the same prompt (should be faster due to cache):")
    start_time = time.time()
    response = llm.get_completion(args.query)
    second_time = time.time() - start_time
    print(f"Response time: {second_time:.4f} seconds")
    print(f"Response: {response[:150]}...")
    
    # Print speed improvement
    if second_time < first_time:
        improvement = (first_time - second_time) / first_time * 100
        print(f"\nSpeed improvement: {improvement:.2f}% faster")
    else:
        print("\nNo speed improvement observed.")

if __name__ == "__main__":
    main() 