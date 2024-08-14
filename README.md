# 🚀 Auto-Cat: Automatic Category Generation


Tired of manually sifting through mountains of unstructured data? Meet Auto-Cat, your AI-powered solution for Automatic Category Generation! 🎉

Auto-Cat isn't just another categorization tool; it's a sophisticated, multi-algorithm approach to understanding and organizing your data. Whether you're dealing with customer feedback, support tickets, or any text-based dataset, Auto-Cat is here to transform chaos into clarity!

## 🌟 Features That Set Auto-Cat Apart

- **Multi-Algorithm Mastery**: KMeans, DBSCAN, and Agglomerative Clustering work in harmony to provide comprehensive insights.
- **Smart Embedding Generation**: Leveraging state-of-the-art language models for deep data understanding.
- **Interactive Visualizations**: Stunning dendrograms, word clouds, and heatmaps that bring your data to life.
- **Comprehensive Reporting**: Get insights that turn raw data into actionable intelligence.
- **Scalability**: From small datasets to big data challenges, Auto-Cat handles it all efficiently.

## 🏃‍♂️ Quick Start

1. Clone this repo: `git clone https://github.com/crypticpy/Auto-Cat.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure a .env file with your `OPENAI_API_KEY`
4. Run the main script: `python main.py`
5. Follow the prompts and watch Auto-Cat work its magic!

## 🧠 The Science Behind Auto-Cat

Auto-Cat represents a cutting-edge approach to automatic category generation, leveraging advanced machine learning techniques and custom-built algorithms to transform unstructured text data into meaningful, actionable categories. Here's an in-depth look at the scientific principles and innovative methods that power Auto-Cat:

## 1. Natural Language Processing (NLP) and Embedding Generation

At the core of Auto-Cat's functionality is its ability to convert text into high-dimensional vector representations, or embeddings. This process relies on state-of-the-art language models, such as OpenAI's text-embedding-3-large.

Key features of our embedding generation process:
- Asynchronous processing for enhanced performance
- Efficient batch handling for large datasets
- Smart caching to minimize redundant API calls
- Real-time progress tracking

These embeddings capture the semantic meaning of the text, allowing for sophisticated analysis beyond simple keyword matching.

## 2. Multi-Algorithm Clustering Approach

Auto-Cat employs a multi-pronged clustering strategy, utilizing three distinct algorithms:

1. **K-Means**: A centroid-based algorithm that aims to partition observations into k clusters.
2. **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise): Excellent for discovering clusters of arbitrary shape and identifying outliers.
3. **Agglomerative Clustering**: A hierarchical clustering approach that builds nested clusters by merging or splitting them successively.

This multi-algorithm approach provides a comprehensive view of the data structure, allowing for cross-validation of results and insights into different aspects of the categorical relationships within the data.

## 3. Optimized Agglomerative Clustering

A standout feature of Auto-Cat is its optimized implementation of agglomerative clustering. This function enhances scalability by:
- Utilizing the 'ward' linkage method for faster, more balanced hierarchies
- Directly applying clustering to embeddings, bypassing the need for a separate distance matrix
- Computing a linkage matrix for further analysis and visualization

## 4. Advanced Category Analysis

Once clusters are formed, Auto-Cat performs in-depth analysis to extract meaningful insights. This analysis includes:
- Extraction of common words within each category
- Selection of representative items for each category
- Robust error handling and edge case management

## 5. Sophisticated Visualization Techniques

Auto-Cat employs advanced visualization methods to make complex data structures interpretable. Our custom functions create detailed dendrograms with:
- Custom node labeling based on category content
- Adjustable detail levels
- High-resolution output for intricate analysis

Additional visualizations include word clouds, heatmaps, and comparative charts for different clustering methods.

## 6. Intelligent Data Preprocessing

Before any analysis occurs, Auto-Cat ensures data quality through intelligent preprocessing. This step handles common text issues like URLs, special characters, and inconsistent formatting, ensuring a clean, standardized dataset for analysis.

## 7. Comprehensive Reporting

Auto-Cat doesn't just analyze data; it communicates results effectively through detailed reports. These reports include:
- Summary statistics
- Visual representations of category distributions
- Detailed breakdowns of category content

## Conclusion

Auto-Cat represents a sophisticated approach to automatic category generation, combining cutting-edge NLP techniques with custom-optimized clustering algorithms. By leveraging multiple clustering methods, advanced visualization techniques, and in-depth category analysis, Auto-Cat provides a robust, scalable solution for deriving meaningful insights from unstructured text data. Its innovative design allows it to handle large datasets efficiently while providing nuanced, actionable categorization results.

## 🔧 How Auto-Cat Works

Auto-Cat is designed to work with CSV files, making it compatible with a wide range of data sources. Here's a step-by-step breakdown of the process:

1. **File Input**: Start by providing a CSV file containing your unstructured data.

2. **Column Selection**: Auto-Cat displays all available columns from your CSV. You choose which columns to include in the analysis, giving you control over the categorization process.

3. **Embedding Generation**: The selected text data is transformed into high-dimensional vector representations using state-of-the-art language models. This step captures the semantic meaning of your data.

4. **Clustering Algorithms**: Auto-Cat applies multiple clustering algorithms (KMeans, DBSCAN, and Agglomerative Clustering) to these embeddings, each offering a unique perspective on your data's structure.

5. **Analysis and Visualization**: The results are analyzed and visualized through various methods, including dendrograms, word clouds, and heatmaps, providing you with comprehensive insights.

## 🧠 Under the Hood

Auto-Cat leverages several sophisticated techniques to deliver accurate and meaningful categorizations:

- **Data Preprocessing**: The `data_ingestion.py` module handles data cleaning, removing duplicates, and preparing text for analysis.

- **Embedding Generation**: Using the `embedding_generation.py` module, Auto-Cat employs advanced NLP models (like OpenAI's text-embedding-3-large) to convert text into dense vector representations.

- **Multi-Algorithm Approach**: The `clustering.py` module implements KMeans, DBSCAN, and Agglomerative Clustering, allowing for comparison and validation of categorization results.

- **Category Analysis**: The `category_analysis.py` module extracts meaningful insights from each cluster, identifying common words and representative items.

- **Visualization**: The `reporting.py` module and the accompanying Jupyter notebook create a variety of visualizations to help you understand and interpret the results.

## 📊 Dive Deeper with Our Jupyter Notebook

For data enthusiasts, we've included a Jupyter notebook (`reporting_notebook.ipynb`) that takes your analysis to the next level. It's packed with advanced visualizations and statistical analyses to give you a comprehensive understanding of your categorized data.

The included capabilities are:

- Generate detailed dendrograms with labeled nodes
- Create word clouds for each clustering method
- Produce heatmaps showing relationships between categories and common words
- Compare the performance of different clustering algorithms
- Analyze the distribution of category sizes


## 📜 License

Auto-Cat is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

