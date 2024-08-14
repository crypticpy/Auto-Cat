# 🚀 Auto-Cat: Automatic Category Generation


## Welcome to the Data Categorization!

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
3. Run the main script: `python main.py`
4. Follow the prompts and watch Auto-Cat work its magic!

## 🧠 The Science Behind Auto-Cat

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

