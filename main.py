# main.py

import asyncio
import logging
import os
from typing import Dict, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from category_analysis import analyze_categories
from clustering import process_clustering
from data_ingestion import load_data
from embedding_generation import generate_embeddings, embeddings_exist, load_embeddings, save_embeddings
from reporting import generate_report, compare_clustering_methods, visualize_hierarchy, generate_word_cloud
from utils import get_user_input, print_summary, validate_file_path, validate_directory

console = Console()
logger = logging.getLogger(__name__)

def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

async def main() -> None:
    try:
        # Get user input and set up logging
        input_file, selected_columns, output_dir, n_clusters = get_user_input()
        validate_file_path(input_file)
        validate_directory(output_dir)
        print_summary(input_file, selected_columns, output_dir, n_clusters)
        log_file = os.path.join(output_dir, "app.log")
        setup_logging(log_file)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn()
        )

        with progress:
            # Data Loading
            load_task = progress.add_task("[green]Loading data...", total=100)
            data = load_data(input_file, selected_columns)
            if data is None or data.empty:
                raise ValueError("No valid data loaded from the input file.")
            progress.update(load_task, completed=100)

            # Embedding Generation
            embed_task = progress.add_task("[yellow]Generating embeddings...", total=100)
            embeddings_file = os.path.join(output_dir, "embeddings.npz")
            if embeddings_exist(embeddings_file):
                embeddings = load_embeddings(embeddings_file)
                if embeddings is None:
                    raise ValueError("Failed to load existing embeddings.")
                logger.info("Loaded existing embeddings")
                progress.update(embed_task, completed=100)
            else:
                def update_embedding_progress(value: float) -> None:
                    progress.update(embed_task, completed=value * 100)

                combined_text = data[selected_columns].astype(str).agg(' '.join, axis=1).tolist()
                embeddings = await generate_embeddings(combined_text, progress_callback=update_embedding_progress)
                if embeddings is None:
                    raise ValueError("Failed to generate embeddings.")
                save_embeddings(embeddings, embeddings_file)
                logger.info("Generated and saved new embeddings")

            # Clustering
            clustering_methods = ["kmeans", "dbscan", "agglomerative"]
            all_clusters: Dict[str, Any] = {}
            linkage_matrix = None
            cluster_task = progress.add_task("[blue]Performing clustering...", total=len(clustering_methods))
            for method in clustering_methods:
                logger.info(f"Starting clustering with method: {method}")
                clusters, linkage = process_clustering(embeddings['embeddings'], method, n_clusters, output_dir)
                if clusters is None:
                    logger.warning(f"Clustering failed for method: {method}")
                    continue
                all_clusters[method] = clusters
                if method == "agglomerative":
                    linkage_matrix = linkage
                logger.info(f"Finished clustering with method: {method}")
                progress.advance(cluster_task)

            # Category Analysis
            analysis_task = progress.add_task("[magenta]Analyzing categories...", total=len(all_clusters))
            all_categories: Dict[str, Dict[int, Dict[str, Any]]] = {}
            for method, clusters in all_clusters.items():
                logger.info(f"Starting category analysis for method: {method}")
                all_categories[method] = analyze_categories(data, clusters, selected_columns)
                logger.info(f"Finished category analysis for method: {method}")
                progress.advance(analysis_task)

            # Reporting
            report_task = progress.add_task("[cyan]Generating reports...",
                                            total=len(all_categories) + 2)  # +2 for comparison and hierarchy
            for method, categories in all_categories.items():
                generate_report(categories, os.path.join(output_dir, f"{method}_report"), selected_columns)
                try:
                    generate_word_cloud(categories, os.path.join(output_dir, f"{method}_report"))
                except Exception as e:
                    logger.warning(f"Failed to generate word cloud for {method}: {str(e)}")
                progress.advance(report_task)

            compare_clustering_methods(all_categories, output_dir)
            progress.advance(report_task)

            if linkage_matrix is not None:
                visualize_hierarchy(linkage_matrix, all_categories['agglomerative'], output_dir)
            progress.advance(report_task)

        logger.info("Analysis complete. Check the output directory for results.")
        console.print("[bold green]Analysis complete. Check the output directory for results.[/bold green]")

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {str(e)}", exc_info=True)
        console.print("[bold red]An unexpected error occurred. Please check the logs.[/bold red]")
    finally:
        console.print("\nPress Enter to exit...")
        input()

if __name__ == "__main__":
    asyncio.run(main())
