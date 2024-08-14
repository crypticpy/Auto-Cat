# embedding_generation.py

import os
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Callable
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import asyncio
import aiohttp
from cachetools import LRUCache

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 200  # Process 200 tickets per batch
NUM_WORKERS = 5  # Number of concurrent workers

embedding_cache = LRUCache(maxsize=100000)

api_call_count = 0


async def generate_embeddings_batch(session: aiohttp.ClientSession, texts: List[str], api_key: str) -> List[
    List[float]]:
    """Generate embeddings for a batch of texts using aiohttp."""
    global api_call_count
    try:
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "input": texts,
            "model": EMBEDDING_MODEL
        }

        async with session.post(url, headers=headers, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            api_call_count += 1
            logger.info(f"API call count: {api_call_count}")

            embeddings = [item['embedding'] for item in result['data']]
            for text, embedding in zip(texts, embeddings):
                embedding_cache[text] = embedding
            return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings batch: {str(e)}")
        raise


async def process_batch(session: aiohttp.ClientSession, batch: List[str], indices: List[int], api_key: str) -> List[
    Dict[str, Any]]:
    """Process a batch of texts and return their embeddings."""
    try:
        embeddings = await generate_embeddings_batch(session, batch, api_key)
        return [{"index": i, "embedding": emb} for i, emb in zip(indices, embeddings)]
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return []


async def generate_embeddings_worker(session: aiohttp.ClientSession, queue: asyncio.Queue, api_key: str,
                                     results: List[Dict[str, Any]]):
    """Worker function to process batches from the queue."""
    while True:
        batch_data = await queue.get()
        if batch_data is None:  # Sentinel value to indicate end of queue
            queue.task_done()
            break
        batch, indices = batch_data
        batch_results = await process_batch(session, batch, indices, api_key)
        results.extend(batch_results)
        queue.task_done()


async def generate_embeddings_async(data: List[str], progress_callback: Optional[Callable[[float], None]] = None) -> \
List[Dict[str, Any]]:
    """Generate embeddings for a list of texts asynchronously using multiple workers."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

    results = []
    queue = asyncio.Queue()

    # Populate the queue with batches
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]
        indices = list(range(i, min(i + BATCH_SIZE, len(data))))
        await queue.put((batch, indices))

    # Add sentinel values to signal the end of the queue
    for _ in range(NUM_WORKERS):
        await queue.put(None)

    async with aiohttp.ClientSession() as session:
        # Create and start the worker tasks
        workers = [asyncio.create_task(generate_embeddings_worker(session, queue, api_key, results)) for _ in
                   range(NUM_WORKERS)]

        # Monitor progress
        total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
        while not queue.empty():
            if progress_callback:
                progress = 1 - (queue.qsize() / total_batches)
                progress_callback(progress)
            await asyncio.sleep(1)  # Check progress every second

        # Wait for all tasks to complete
        await queue.join()
        await asyncio.gather(*workers)

    # Sort results by index to maintain original order
    results.sort(key=lambda x: x["index"])
    logger.info(f"Generated embeddings for {len(results)} items")
    return [r["embedding"] for r in results]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_embeddings(data: List[str], progress_callback: Optional[Callable[[float], None]] = None) -> Optional[Dict[str, Any]]:
    """
    Generate embeddings for the input data using OpenAI's API.

    Args:
    data (List[str]): List of texts to generate embeddings for.
    progress_callback (Callable[[float], None]): Optional callback function to update progress.

    Returns:
    Optional[Dict[str, Any]]: Dictionary containing embeddings, ids, and metadata, or None if an error occurs.
    """
    try:
        logger.info(f"Starting embedding generation for {len(data)} items")
        embeddings = await generate_embeddings_async(data, progress_callback=progress_callback)

        if not embeddings:
            logger.warning("No embeddings generated")
            return None

        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return {
            'embeddings': np.array(embeddings),
            'ids': list(range(len(embeddings))),
            'metadatas': [{"text": text[:100]} for text in data]  # Truncate metadata to first 100 characters
        }

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return None

def embeddings_exist(file_path: str) -> bool:
    """Check if embeddings file exists."""
    return os.path.exists(file_path)

def save_embeddings(embeddings: Dict[str, Any], file_path: str) -> None:
    """Save embeddings to a file."""
    try:
        np.savez_compressed(file_path,
                            embeddings=embeddings['embeddings'],
                            ids=embeddings['ids'],
                            metadatas=embeddings['metadatas'])
        logger.info(f"Embeddings saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")

def load_embeddings(file_path: str) -> Optional[Dict[str, Any]]:
    """Load embeddings from a file."""
    try:
        with np.load(file_path, allow_pickle=True) as data:
            return {
                'embeddings': data['embeddings'],
                'ids': data['ids'].tolist(),
                'metadatas': data['metadatas'].tolist()
            }
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        return None
