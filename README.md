# PDF Summarization and Topic Modeling System

## Overview
This project extracts text from PDF files, summarizes them using natural language processing models, and applies topic modeling with BERTopic. It leverages libraries such as **PyPDF2**, **Transformers**, **Sentence Transformers**, and **BERTopic** to efficiently process research papers.

## Features
- **PDF Text Extraction:** Extracts text from PDFs using PyPDF2 with error handling.
- **Text Cleaning:** Removes unnecessary content such as tables of contents, references, etc.
- **Text Summarization:** Uses a transformer model (T5-small) to generate summaries for each document.
- **Topic Modeling:** Applies BERTopic along with UMAP and HDBSCAN to cluster and identify topics across papers.
- **Parallel Processing:** Utilizes ThreadPoolExecutor and ProcessPoolExecutor to speed up PDF processing and summarization.
- **Structured Output:** Saves summaries and topic information in a well-organized text file.

## Installation

### Prerequisites
Ensure you have Python 3.7 or higher installed.

### Steps
1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/pdf-summarization.git
   cd pdf-summarization

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt

3. **Download NLTK Resources:**
   ```sh
   python -c "import nltk; nltk.download('punkt')"


### Usage
- **Prepare your PDFs:** Place your PDF files in a folder named papers in the project directory.
- **Run the final.py script:** Execute the script to extract text, generate summaries, and perform topic modeling.
- **Output:** The summaries and topic clustering details will be saved in summaries.txt.



