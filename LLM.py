"""
LLM.py - Prompt Caching Implementations for Different LLM Providers

This module provides implementations of prompt caching for different LLM providers:
- OpenAI
- Google (Gemini)
- Anthropic (Claude)

The goal is to optimize LLM response time by caching common prompts and their responses.
"""

import os
import logging
import hashlib
import json
from typing import Dict, List, Optional, Any, Union, Callable
import time
from functools import lru_cache
from pathlib import Path

# For OpenAI
from openai import OpenAI

# For Google (Gemini)
import google.generativeai as genai
from google.ai import generativelanguage as glm

# For Anthropic
import anthropic # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------
# Utility Functions
# ---------------------------

def get_cache_dir() -> Path:
    """
    Creates and returns the cache directory path.
    
    Returns:
        Path: Path to the cache directory
    """
    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def generate_cache_key(prompt: str, model: str) -> str:
    """
    Generates a unique cache key based on the prompt and model.
    
    Args:
        prompt (str): The prompt text
        model (str): The model identifier
        
    Returns:
        str: A unique hash key
    """
    # Create a unique hash from the prompt and model
    combined = f"{prompt}:{model}"
    return hashlib.md5(combined.encode()).hexdigest()

def load_cache_data() -> Dict[str, Any]:
    """
    Loads cache data from the cache file.
    
    Returns:
        Dict[str, Any]: Dictionary of cached data
    """
    cache_file = get_cache_dir() / "prompt_cache.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return {}
    return {}

def save_cache_data(cache_data: Dict[str, Any]) -> None:
    """
    Saves cache data to the cache file.
    
    Args:
        cache_data (Dict[str, Any]): Dictionary of cached data
    """
    cache_file = get_cache_dir() / "prompt_cache.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

# ---------------------------
# OpenAI Implementation
# ---------------------------

class OpenAIWithCache:
    """
    A class that implements prompt caching for OpenAI models.
    
    This implementation uses a combination of in-memory LRU cache and
    file-based persistent cache to speed up repeated prompts.
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini", 
        api_key: Optional[str] = None,
        max_cache_size: int = 100,
        temperature: float = 0.0  # Low temperature for more consistent caching
    ):
        """
        Initialize the OpenAI client with caching capabilities.
        
        Args:
            model (str): The OpenAI model to use
            api_key (Optional[str]): OpenAI API key (defaults to env var)
            max_cache_size (int): Maximum size of the in-memory LRU cache
            temperature (float): Temperature parameter for generation (lower is more deterministic)
        """
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        
        # Load cache from disk
        self.disk_cache = load_cache_data()
        
        # Configure LRU cache decorator
        self.max_cache_size = max_cache_size
        
        # Apply LRU cache to the completion method
        self._cached_completion = lru_cache(maxsize=max_cache_size)(self._actual_completion)
    
    def _actual_completion(self, cache_key: str, prompt: str) -> str:
        """
        The actual method that calls the OpenAI API.
        
        Args:
            cache_key (str): A unique key for caching
            prompt (str): The prompt to send to the model
            
        Returns:
            str: The model's response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content
    
    def get_completion(self, prompt: str) -> str:
        """
        Get a completion for the given prompt, using cache when available.
        
        Args:
            prompt (str): The prompt to get a completion for
            
        Returns:
            str: The model's response
        """
        start_time = time.time()
        cache_key = generate_cache_key(prompt, self.model)
        
        # Check disk cache first
        if cache_key in self.disk_cache:
            logger.info(f"Disk cache hit for prompt: {prompt[:50]}...")
            response = self.disk_cache[cache_key]
            logger.info(f"Cache retrieval time: {time.time() - start_time:.4f}s")
            return response
        
        # Then try memory cache (LRU)
        try:
            response = self._cached_completion(cache_key, prompt)
            # Save to disk cache
            self.disk_cache[cache_key] = response
            save_cache_data(self.disk_cache)
            logger.info(f"Memory cache hit or API call completed in {time.time() - start_time:.4f}s")
            return response
        except Exception as e:
            logger.error(f"Error in get_completion: {e}")
            # Fall back to direct API call if caching fails
            response = self._actual_completion(cache_key, prompt)
            return response
    
    def preload_cache(self, prompts: List[str]) -> None:
        """
        Preload the cache with a list of prompts.
        
        Args:
            prompts (List[str]): List of prompts to preload
        """
        for prompt in prompts:
            logger.info(f"Preloading cache for prompt: {prompt[:50]}...")
            self.get_completion(prompt)

# ---------------------------
# Google (Gemini) Implementation
# ---------------------------

class GeminiWithCache:
    """
    A class that implements prompt caching for Google's Gemini models.
    
    This implementation handles caching to optimize response time for
    repeated prompts.
    """
    
    def __init__(
        self, 
        model: str = "gemini-2.0-flash", 
        api_key: Optional[str] = None,
        max_cache_size: int = 100,
        temperature: float = 0.0  # Low temperature for more consistent caching
    ):
        """
        Initialize the Gemini client with caching capabilities.
        
        Args:
            model (str): The Gemini model to use
            api_key (Optional[str]): Google API key (defaults to env var)
            max_cache_size (int): Maximum size of the in-memory LRU cache
            temperature (float): Temperature parameter for generation
        """
        # Initialize Google Generative AI client
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        genai.configure(api_key=api_key)
        
        self.model = model
        self.temperature = temperature
        self.generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            # Can add other generation parameters here
        )
        
        # Set up Google model
        self.gemini_model = genai.GenerativeModel(model_name=model)
        
        # Load cache from disk
        self.disk_cache = load_cache_data()
        
        # Configure LRU cache decorator
        self.max_cache_size = max_cache_size
        
        # Apply LRU cache to the completion method
        self._cached_completion = lru_cache(maxsize=max_cache_size)(self._actual_completion)
    
    def _actual_completion(self, cache_key: str, prompt: str) -> str:
        """
        The actual method that calls the Gemini API.
        
        Args:
            cache_key (str): A unique key for caching
            prompt (str): The prompt to send to the model
            
        Returns:
            str: The model's response
        """
        response = self.gemini_model.generate_content(
            prompt,
            generation_config=self.generation_config
        )
        return response.text
    
    def get_completion(self, prompt: str) -> str:
        """
        Get a completion for the given prompt, using cache when available.
        
        Args:
            prompt (str): The prompt to get a completion for
            
        Returns:
            str: The model's response
        """
        start_time = time.time()
        cache_key = generate_cache_key(prompt, self.model)
        
        # Check disk cache first
        if cache_key in self.disk_cache:
            logger.info(f"Disk cache hit for prompt: {prompt[:50]}...")
            response = self.disk_cache[cache_key]
            logger.info(f"Cache retrieval time: {time.time() - start_time:.4f}s")
            return response
        
        # Then try memory cache (LRU)
        try:
            response = self._cached_completion(cache_key, prompt)
            # Save to disk cache
            self.disk_cache[cache_key] = response
            save_cache_data(self.disk_cache)
            logger.info(f"Memory cache hit or API call completed in {time.time() - start_time:.4f}s")
            return response
        except Exception as e:
            logger.error(f"Error in get_completion: {e}")
            # Fall back to direct API call if caching fails
            response = self._actual_completion(cache_key, prompt)
            return response
    
    def preload_cache(self, prompts: List[str]) -> None:
        """
        Preload the cache with a list of prompts.
        
        Args:
            prompts (List[str]): List of prompts to preload
        """
        for prompt in prompts:
            logger.info(f"Preloading cache for prompt: {prompt[:50]}...")
            self.get_completion(prompt)

# ---------------------------
# Anthropic (Claude) Implementation
# ---------------------------

class ClaudeWithCache:
    """
    A class that implements prompt caching for Anthropic's Claude models.
    
    This implementation uses Anthropic's prompt caching mechanism along with
    our own local caching for redundancy and improved performance.
    """
    
    def __init__(
        self, 
        model: str = "claude-3-sonnet-20240229", 
        api_key: Optional[str] = None,
        max_cache_size: int = 100,
        temperature: float = 0.0  # Low temperature for more consistent caching
    ):
        """
        Initialize the Claude client with caching capabilities.
        
        Args:
            model (str): The Claude model to use
            api_key (Optional[str]): Anthropic API key (defaults to env var)
            max_cache_size (int): Maximum size of the in-memory LRU cache
            temperature (float): Temperature parameter for generation
        """
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature
        
        # Load cache from disk
        self.disk_cache = load_cache_data()
        
        # Configure LRU cache decorator
        self.max_cache_size = max_cache_size
        
        # Apply LRU cache to the completion method
        self._cached_completion = lru_cache(maxsize=max_cache_size)(self._actual_completion)
    
    def _actual_completion(self, cache_key: str, prompt: str) -> str:
        """
        The actual method that calls the Claude API, using Anthropic's built-in caching.
        
        Args:
            cache_key (str): A unique key for caching
            prompt (str): The prompt to send to the model
            
        Returns:
            str: The model's response
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ],
            # Enable Anthropic's built-in caching
            extra_headers={
                # Set a stable client ID for consistent caching across requests
                "X-Anthropic-Client-Id": "my-app-id",
                "X-Anthropic-Cache-Control": "max-age=3600",  # Cache for 1 hour
            }
        )
        return response.content[0].text
    
    def get_completion(self, prompt: str) -> str:
        """
        Get a completion for the given prompt, using cache when available.
        
        Args:
            prompt (str): The prompt to get a completion for
            
        Returns:
            str: The model's response
        """
        start_time = time.time()
        cache_key = generate_cache_key(prompt, self.model)
        
        # Check disk cache first
        if cache_key in self.disk_cache:
            logger.info(f"Disk cache hit for prompt: {prompt[:50]}...")
            response = self.disk_cache[cache_key]
            logger.info(f"Cache retrieval time: {time.time() - start_time:.4f}s")
            return response
        
        # Then try memory cache (LRU) or API with Anthropic's caching
        try:
            response = self._cached_completion(cache_key, prompt)
            # Save to disk cache
            self.disk_cache[cache_key] = response
            save_cache_data(self.disk_cache)
            logger.info(f"Memory cache hit or API call completed in {time.time() - start_time:.4f}s")
            return response
        except Exception as e:
            logger.error(f"Error in get_completion: {e}")
            # Fall back to direct API call if caching fails
            response = self._actual_completion(cache_key, prompt)
            return response
    
    def preload_cache(self, prompts: List[str]) -> None:
        """
        Preload the cache with a list of prompts.
        
        Args:
            prompts (List[str]): List of prompts to preload
        """
        for prompt in prompts:
            logger.info(f"Preloading cache for prompt: {prompt[:50]}...")
            self.get_completion(prompt)

# ---------------------------
# Helper function to generate prompts from the PDF data
# ---------------------------

def generate_prompts_from_data(pdf_dir: str, max_tokens: int = 10000) -> List[str]:
    """
    Generates a list of prompts from the PDF data, ensuring the total
    token count stays within the specified limit.
    
    Args:
        pdf_dir (str): Directory containing PDF files
        max_tokens (int): Maximum token count to include
        
    Returns:
        List[str]: List of prompts generated from the data
    """
    from langchain_community.document_loaders import PyMuPDFLoader
    from chunking import Chunker
    
    prompts = []
    total_tokens = 0
    token_estimate_ratio = 1.3  # Rough estimate of tokens per character
    
    # Create a list of PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    # Process each PDF file
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_dir, pdf_file)
        
        # Load the document
        document_loader = PyMuPDFLoader(file_path)
        documents = document_loader.load()
        
        # Split the document into chunks
        chunks = Chunker().chunk(documents)
        
        # Process each chunk
        for chunk in chunks:
            # Estimate token count for this chunk
            chunk_tokens = len(chunk.page_content) / token_estimate_ratio
            
            # If adding this chunk would exceed the token limit, stop
            if total_tokens + chunk_tokens > max_tokens:
                break
                
            # Create a prompt from this chunk
            prompt = f"Based on the following information: {chunk.page_content}\n\nProvide a concise summary."
            prompts.append(prompt)
            
            # Update token count
            total_tokens += chunk_tokens
        
        # If we've reached the token limit, stop processing files
        if total_tokens >= max_tokens:
            break
    
    logger.info(f"Generated {len(prompts)} prompts with approximately {total_tokens:.0f} tokens")
    return prompts

# ---------------------------
# Usage example
# ---------------------------

def preload_ono_data_to_cache(llm_provider: str = "openai", max_tokens: int = 10000) -> None:
    """
    Preloads the Ono data to the LLM cache for faster responses.
    
    Args:
        llm_provider (str): Which LLM provider to use ('openai', 'google', or 'anthropic')
        max_tokens (int): Maximum token count to include
    """
    # Generate prompts from the Ono data
    prompts = generate_prompts_from_data("Ono data", max_tokens)
    
    # Preload the cache based on the selected provider
    if llm_provider.lower() == "openai":
        llm = OpenAIWithCache()
        llm.preload_cache(prompts)
    elif llm_provider.lower() == "google":
        llm = GeminiWithCache()
        llm.preload_cache(prompts)
    elif llm_provider.lower() == "anthropic":
        llm = ClaudeWithCache()
        llm.preload_cache(prompts)
    else:
        logger.error(f"Unknown LLM provider: {llm_provider}")
        return
    
    logger.info(f"Successfully preloaded {len(prompts)} prompts to {llm_provider} cache")

if __name__ == "__main__":
    # Example usage
    # preload_ono_data_to_cache("openai")
    pass
