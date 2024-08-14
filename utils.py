# utils.py
import os
import pandas as pd
from typing import List, Tuple, Optional
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
import multiprocessing

console = Console()


def find_csv_files(directory: str = '.') -> List[str]:
    """
    Find all CSV files in the given directory.

    Args:
    directory (str): Directory to search for CSV files.

    Returns:
    List[str]: List of CSV file names found in the directory.
    """
    return [f for f in os.listdir(directory) if f.endswith('.csv')]


def get_user_input() -> Tuple[str, List[str], str, int]:
    """
    Get user input for file selection, column selection, output directory, and number of clusters.

    Returns: Tuple[str, List[str], str, int]: Tuple containing input file path, selected columns, output directory,
    and number of clusters.
    """
    console.print("[bold]Welcome to the ServiceNow Ticket Categorization System![/bold]")

    csv_files = find_csv_files()
    if not csv_files:
        console.print("[red]No CSV files found in the current directory.[/red]")
        raise FileNotFoundError("No CSV files available")

    console.print("Available CSV files:")
    for i, file in enumerate(csv_files, 1):
        console.print(f"{i}. {file}")

    while True:
        choice = IntPrompt.ask("Enter the number of the CSV file you want to use", default=1)
        if 1 <= choice <= len(csv_files):
            input_file = csv_files[choice - 1]
            break
        console.print("[red]Invalid choice. Please try again.[/red]")

    # Display columns and let user select by numbers
    try:
        df = pd.read_csv(input_file, nrows=0)
    except Exception as e:
        console.print(f"[red]Error reading CSV file: {str(e)}[/red]")
        raise

    columns = df.columns.tolist()

    console.print("Available columns:")
    column_panels = [Panel(f"{i + 1}. {col}", expand=False) for i, col in enumerate(columns)]
    console.print(Columns(column_panels, equal=True, expand=False))

    while True:
        selected_indices = Prompt.ask("Enter the numbers of the columns you want to use (comma-separated)")
        try:
            selected_indices = [int(i.strip()) for i in selected_indices.split(',')]
            if all(1 <= i <= len(columns) for i in selected_indices):
                selected_columns = [columns[i - 1] for i in selected_indices]
                break
            else:
                console.print("[red]Invalid column number(s). Please try again.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter comma-separated numbers.[/red]")

    output_dir = Prompt.ask("Enter the path for the output directory", default="./output")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        console.print(f"[red]Error creating output directory: {str(e)}[/red]")
        raise

    n_clusters = IntPrompt.ask("Enter the number of clusters for KMeans", default=10)

    return input_file, selected_columns, output_dir, n_clusters


def get_embedding_batch_size() -> int:
    """
    Get the batch size for embedding generation from user input.

    Returns:
    int: Batch size for embedding generation.
    """
    return IntPrompt.ask("Enter the batch size for embedding generation", default=100)


def get_num_workers() -> int:
    """
    Get the number of worker processes for parallel processing.

    Returns:
    int: Number of worker processes.
    """
    max_workers = multiprocessing.cpu_count()
    return IntPrompt.ask(f"Enter the number of worker processes (max {max_workers})", default=max_workers)


def validate_file_path(file_path: str) -> None:
    """
    Validate if a file exists.

    Args:
    file_path (str): Path to the file.

    Raises:
    FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


def validate_directory(directory: str) -> None:
    """
    Validate if a directory exists, create it if it doesn't.

    Args:
    directory (str): Path to the directory.

    Raises:
    OSError: If the directory cannot be created.
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            raise OSError(f"Error creating directory {directory}: {str(e)}") from e


def print_summary(input_file: str, selected_columns: List[str], output_dir: str, n_clusters: int) -> None:
    """
    Print a summary of the user's input.

    Args:
    input_file (str): Path to the input CSV file.
    selected_columns (List[str]): List of selected column names.
    output_dir (str): Path to the output directory.
    n_clusters (int): Number of clusters for KMeans.
    """
    console.print("\n[bold]Summary of your choices:[/bold]")
    console.print(f"Input file: {input_file}")
    console.print(f"Selected columns: {', '.join(selected_columns)}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Number of clusters: {n_clusters}")
    console.print("\nPress Enter to continue or Ctrl+C to abort.")
    input()


def create_progress_bar(description: str) -> Progress:
    """
    Create a progress bar with a given description.

    Args:
    description (str): Description of the progress bar.

    Returns:
    Progress: A Rich progress bar object.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
    )


def confirm_action(message: str) -> bool:
    """
    Ask the user to confirm an action.

    Args:
    message (str): The confirmation message to display.

    Returns:
    bool: True if the user confirms, False otherwise.
    """
    return Prompt.ask(message, choices=["y", "n"], default="y") == "y"


def get_file_info(file_path: str) -> Optional[dict]:
    """
    Get information about a file.

    Args:
    file_path (str): Path to the file.

    Returns:
    Optional[dict]: A dictionary containing file information, or None if the file doesn't exist.
    """
    if not os.path.exists(file_path):
        return None

    return {
        "size": os.path.getsize(file_path),
        "last_modified": os.path.getmtime(file_path),
        "is_directory": os.path.isdir(file_path)
    }


if __name__ == "__main__":
    # Example usage
    input_file, selected_columns, output_dir, n_clusters = get_user_input()
    print_summary(input_file, selected_columns, output_dir, n_clusters)
