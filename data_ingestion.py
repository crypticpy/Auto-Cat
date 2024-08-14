# data_ingestion.py
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
import re

logger = logging.getLogger(__name__)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows based on the full raw text of each row.

    Args:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: Dataframe with duplicates removed.
    """
    initial_count = len(df)

    # Convert each row to a string representation
    df['full_text'] = df.astype(str).agg(' '.join, axis=1)

    # Remove duplicates based on the full text
    df_no_duplicates = df.drop_duplicates(subset=['full_text'], keep='first')

    # Remove the temporary 'full_text' column
    df_no_duplicates = df_no_duplicates.drop(columns=['full_text'])

    removed_count = initial_count - len(df_no_duplicates)

    logger.info(f"Removed {removed_count} duplicate rows")

    return df_no_duplicates


def preprocess_text(text: str) -> str:
    """
    Preprocess a single text string by removing URLs, special characters, and lowercasing.
    """
    # Remove URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub('', text)
    # Remove special characters and lowercase
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())


def preprocess_text_column(series: pd.Series) -> pd.Series:
    """
    Preprocess a text column by removing URLs, special characters, and lowercasing.
    """
    return series.astype(str).apply(preprocess_text)


def load_data(file_path: str, selected_columns: List[str]) -> Optional[pd.DataFrame]:
    """
    Load and preprocess data from a CSV file.

    Args:
    file_path (str): Path to the CSV file.
    selected_columns (List[str]): List of column names to load.

    Returns:
    Optional[pd.DataFrame]: Loaded and preprocessed data, or None if an error occurs.
    """
    try:
        # Validate inputs
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string")
        if not isinstance(selected_columns, list) or not all(isinstance(col, str) for col in selected_columns):
            raise TypeError("selected_columns must be a list of strings")

        # Read the CSV file
        df = pd.read_csv(file_path, usecols=selected_columns, low_memory=False)

        # Log initial data load
        logger.info(f"Loaded {len(df)} records from {file_path}")

        # Check if any selected columns are missing
        missing_columns = set(selected_columns) - set(df.columns)
        if missing_columns:
            logger.warning(f"The following columns are missing from the CSV: {missing_columns}")
            selected_columns = [col for col in selected_columns if col in df.columns]

        # Remove duplicates
        df = remove_duplicates(df)

        # Remove rows with missing values in selected columns
        df_clean = df.dropna(subset=selected_columns)

        # Log data cleaning results
        rows_removed = len(df) - len(df_clean)
        logger.info(f"Removed {rows_removed} rows with missing values")
        logger.info(f"{len(df_clean)} records remaining after cleaning")

        # Verify that we have data left after cleaning
        if df_clean.empty:
            logger.error("No data remaining after removing rows with missing values")
            return None

        # Basic data type conversions and handling
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = preprocess_text_column(df_clean[col])
            elif pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(0)

        return df_clean

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {file_path}")
    except pd.errors.ParserError:
        logger.error(f"Error parsing CSV file: {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")

    return None


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of the dataframe.

    Args:
    df (pd.DataFrame): The dataframe to summarize.

    Returns:
    Dict[str, Any]: A dictionary containing summary statistics.
    """
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "unique_values": {col: df[col].nunique() for col in df.columns}
    }


def sample_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return a sample of n rows from the dataframe.

    Args:
    df (pd.DataFrame): The dataframe to sample from.
    n (int): The number of rows to sample.

    Returns:
    pd.DataFrame: A sample of n rows from the dataframe.
    """
    return df.sample(n=min(n, len(df)))


if __name__ == "__main__":
    # Example usage
    file_path = "example.csv"
    columns = ["column1", "column2", "column3"]
    df = load_data(file_path, columns)
    if df is not None:
        print(get_data_summary(df))
        print(sample_data(df))
