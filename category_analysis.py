# category_analysis.py
from collections import defaultdict
import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    """Preprocess the text by lowercasing and removing special characters."""
    text = text.lower()
    return re.sub(r'[^a-zA-Z\s]', '', text)

@lru_cache(maxsize=1000)
def get_common_words_cached(text: str, n: int = 10) -> List[str]:
    """
    Extract common words from a text string (cached version).

    Args:
    text (str): Text string to analyze.
    n (int): Number of top common words to return.

    Returns:
    List[str]: List of the n most common words.
    """
    try:
        if not text.strip():
            logger.warning("Empty text provided for common word analysis")
            return []

        preprocessed_text = preprocess_text(text)

        custom_stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']

        vectorizer = CountVectorizer(stop_words=custom_stop_words, token_pattern=r'\b[a-zA-Z]{3,}\b')
        word_counts = vectorizer.fit_transform([preprocessed_text])

        words = vectorizer.get_feature_names_out()
        counts = word_counts.toarray()[0]

        word_freq = sorted(zip(words, counts), key=lambda x: x[1], reverse=True)
        top_words = [word for word, count in word_freq[:n] if count > 0]

        if not top_words:
            logger.warning("No common words found after processing")

        return top_words

    except Exception as e:
        logger.error(f"Error getting common words: {str(e)}")
        return []

def get_common_words(texts: List[str], n: int = 10) -> List[str]:
    """
    Extract common words from a list of texts.

    Args:
    texts (List[str]): List of text strings to analyze.
    n (int): Number of top common words to return.

    Returns:
    List[str]: List of the n most common words.
    """
    try:
        if not texts:
            logger.warning("No texts provided for common word analysis")
            return []

        all_text = ' '.join(str(text) for text in texts if text)
        return get_common_words_cached(all_text, n)

    except Exception as e:
        logger.error(f"Error getting common words: {str(e)}")
        return []

def get_representative_items(items: List[Dict[str, Any]], selected_columns: List[str], n: int = 5) -> List[str]:
    """
    Select representative items based on the length of their combined text.

    Args:
    items (List[Dict[str, Any]]): List of items (each a dictionary of column values).
    selected_columns (List[str]): List of column names to use.
    n (int): Number of representative items to return.

    Returns:
    List[str]: List of n representative item texts.
    """
    try:
        if not items:
            logger.warning("No items provided for representative item selection")
            return []

        combined_texts = [
            ' '.join(str(item.get(col, '')) for col in selected_columns)
            for item in items
        ]

        sorted_items = sorted(combined_texts, key=len, reverse=True)
        return [item[:1000] for item in sorted_items[:n]]  # Truncate to 1000 characters

    except Exception as e:
        logger.error(f"Error getting representative items: {str(e)}")
        return []

def analyze_categories(data: pd.DataFrame, clusters: np.ndarray, selected_columns: List[str],
                       n_common_words: int = 10, n_representative_items: int = 5) -> Dict[int, Dict[str, Any]]:
    """
    Analyze the resulting clusters to derive meaningful categories.

    Args:
    data (pd.DataFrame): The original dataset.
    clusters (np.ndarray): Array of cluster labels for each data point.
    selected_columns (List[str]): List of column names used for analysis.
    n_common_words (int): Number of common words to extract for each category.
    n_representative_items (int): Number of representative items to select for each category.

    Returns:
    Dict[int, Dict[str, Any]]: Dictionary of analyzed categories.
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(clusters, np.ndarray):
            raise TypeError("clusters must be a numpy array")
        if not isinstance(selected_columns, list) or not all(isinstance(col, str) for col in selected_columns):
            raise TypeError("selected_columns must be a list of strings")
        if any(col not in data.columns for col in selected_columns):
            raise ValueError("Not all selected columns are present in the dataframe")

        categories = defaultdict(list)
        for i, cluster in enumerate(clusters):
            categories[cluster].append(data.iloc[i])

        analyzed_categories = {}
        for cluster, items in categories.items():
            combined_texts = [
                ' '.join(str(item[col]) for col in selected_columns if pd.notna(item[col]))
                for item in items
            ]

            analyzed_categories[cluster] = {
                "size": len(items),
                "common_words": get_common_words(combined_texts, n=n_common_words),
                "representative_items": get_representative_items(items, selected_columns, n=n_representative_items)
            }

        logger.info(f"Analyzed {len(analyzed_categories)} categories")
        return analyzed_categories

    except Exception as e:
        logger.error(f"Error analyzing categories: {str(e)}")
        raise

def process_items_in_batches(items: List[Any], batch_size: int = 1000) -> List[List[Any]]:
    """Process items in batches to handle large datasets."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
