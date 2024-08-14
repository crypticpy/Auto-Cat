# reporting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import os
from typing import Dict, Any, List
from scipy.cluster.hierarchy import dendrogram
import numpy as np

logger = logging.getLogger(__name__)


def generate_report(categories: Dict[int, Dict[str, Any]], output_path: str, selected_columns: List[str]) -> None:
    """
    Generate comprehensive reports and visualizations of the categorization results.

    Args:
    categories (Dict[int, Dict[str, Any]]): Dictionary of analyzed categories.
    output_path (str): Path to save the generated reports and visualizations.
    selected_columns (List[str]): List of column names used in the analysis.
    """
    try:
        if not isinstance(categories, dict):
            raise TypeError("categories must be a dictionary")
        if not isinstance(output_path, str):
            raise TypeError("output_path must be a string")
        if not isinstance(selected_columns, list) or not all(isinstance(col, str) for col in selected_columns):
            raise TypeError("selected_columns must be a list of strings")

        os.makedirs(output_path, exist_ok=True)

        # Generate summary statistics
        summary = pd.DataFrame([(k, v['size']) for k, v in categories.items()], columns=['Category', 'Size'])
        summary.to_csv(os.path.join(output_path, "category_summary.csv"), index=False)

        # Generate visualizations
        plt.figure(figsize=(12, 6))
        summary['Category'] = summary['Category'].astype(str)
        sns.barplot(x='Category', y='Size', data=summary)
        plt.title("Item Distribution Across Categories")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "category_distribution.png"))
        plt.close()

        # Generate detailed report
        with open(os.path.join(output_path, "detailed_report.txt"), "w") as f:
            for category, info in categories.items():
                f.write(f"Category {category}:\n")
                f.write(f"  Size: {info['size']}\n")
                f.write(f"  Common Words: {', '.join(info['common_words'])}\n")
                f.write("  Representative Items:\n")
                for item in info['representative_items']:
                    f.write(f"    - {item[:100]}...\n")  # Truncate long descriptions
                f.write("\n")

        logger.info(f"Generated reports in {output_path}")
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise


def compare_clustering_methods(all_categories: Dict[str, Dict[int, Dict[str, Any]]], output_path: str) -> None:
    """
    Compare the results of different clustering methods.

    Args:
    all_categories (Dict[str, Dict[int, Dict[str, Any]]]): Dictionary of categories for each clustering method.
    output_path (str): Path to save the comparison results.
    """
    try:
        if not isinstance(all_categories, dict):
            raise TypeError("all_categories must be a dictionary")
        if not isinstance(output_path, str):
            raise TypeError("output_path must be a string")

        comparison = {}
        for method, categories in all_categories.items():
            num_categories = len(categories)
            if num_categories > 0:
                total_size = sum(cat['size'] for cat in categories.values())
                avg_size = total_size / num_categories
                max_size = max((cat['size'] for cat in categories.values()), default=0)
                min_size = min((cat['size'] for cat in categories.values()), default=0)
            else:
                avg_size = max_size = min_size = 0

            comparison[method] = {
                "num_categories": num_categories,
                "avg_category_size": avg_size,
                "max_category_size": max_size,
                "min_category_size": min_size,
            }

        df = pd.DataFrame(comparison).T
        df.to_csv(os.path.join(output_path, "clustering_comparison.csv"))

        if not df.empty:
            plt.figure(figsize=(12, 6))
            df.plot(kind='bar', y=['num_categories', 'avg_category_size'], ax=plt.gca())
            plt.title("Comparison of Clustering Methods")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "clustering_comparison.png"))
            plt.close()
        else:
            logger.warning("No data available for clustering comparison plot")

        logger.info(f"Generated clustering comparison in {output_path}")
    except Exception as e:
        logger.error(f"Error comparing clustering methods: {str(e)}")
        raise


def visualize_hierarchy(linkage_matrix: np.ndarray, categories: Dict[int, Dict[str, Any]], output_path: str) -> None:
    try:
        plt.figure(figsize=(20, 10))

        # Generate labels based on the linkage matrix size
        n_samples = linkage_matrix.shape[0] + 1
        labels = [f"Category {i}" for i in range(n_samples)]

        dendrogram(
            linkage_matrix,
            leaf_rotation=90,
            leaf_font_size=8,
            labels=labels
        )
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Category")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "hierarchy_dendrogram.png"), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Generated hierarchical clustering dendrogram with category labels")
    except Exception as e:
        logger.error(f"Error visualizing hierarchy: {str(e)}")
        logger.error(f"Linkage matrix shape: {linkage_matrix.shape}")
        logger.error(f"Number of categories: {len(categories)}")
        raise


def generate_word_cloud(categories: Dict[int, Dict[str, Any]], output_path: str) -> None:
    """
    Generate a word cloud visualization for all categories combined.

    Args:
    categories (Dict[int, Dict[str, Any]]): Dictionary of analyzed categories.
    output_path (str): Path to save the word cloud visualization.
    """
    try:
        from wordcloud import WordCloud

        if not isinstance(categories, dict):
            raise TypeError("categories must be a dictionary")
        if not isinstance(output_path, str):
            raise TypeError("output_path must be a string")

        all_words = ' '.join([' '.join(cat['common_words']) for cat in categories.values() if cat['common_words']])

        if not all_words:
            logger.warning("No common words found across categories. Skipping word cloud generation.")
            return

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(output_path, "word_cloud.png"))
        plt.close()

        logger.info(f"Generated word cloud in {output_path}")
    except ImportError:
        logger.warning("WordCloud not installed. Skipping word cloud generation.")
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
