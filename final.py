import os
import re
import nltk
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import PyPDF2
from umap import UMAP
from hdbscan import HDBSCAN
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with error handling"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
    return text

def clean_text(text):
    """Clean extracted text more efficiently"""
    text = re.sub(r"(Table of Contents|References|Bibliography).*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def process_pdf(file_path):
    """Process a single PDF file"""
    raw_text = extract_text_from_pdf(file_path)
    cleaned_text = clean_text(raw_text)
    return os.path.basename(file_path), cleaned_text if len(cleaned_text.split()) > 100 else None

def load_papers_from_pdfs(directory):
    """Load papers using parallel processing"""
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    papers = {}
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_pdf, pdf_files), total=len(pdf_files), desc="Loading PDFs"))
    
    for result in results:
        if result and result[1]:
            papers[result[0]] = result[1]
    return papers

def split_text_into_chunks(text, max_chunk_length=512):
    """Split text into chunks more efficiently"""
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk = [], []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) > max_chunk_length:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [sentence], len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_text(text, summarizer, max_length=50, min_length=20, chunk_size=512):
    """Summarize text with batch processing for efficiency"""
    chunks = split_text_into_chunks(text, chunk_size)
    summaries = summarizer(chunks, max_length=max_length, min_length=min_length, do_sample=False)
    return " ".join([s['summary_text'] for s in summaries if 'summary_text' in s])

def process_paper_summary(args):
    """Process a single paper for summarization"""
    paper_name, paper_text, summarizer = args
    try:
        summary = summarize_text(paper_text, summarizer)
        return paper_name, summary
    except Exception as e:
        logger.error(f"Error summarizing {paper_name}: {e}")
        return paper_name, f"Error: {str(e)}"

def main():
    start_time = time.time()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    pdf_directory = "papers"
    logger.info(f"Loading papers from {pdf_directory}")
    papers = load_papers_from_pdfs(pdf_directory)
    
    if len(papers) < 2:
        raise ValueError("Not enough valid papers. Ensure at least two non-empty PDFs are available.")
    
    logger.info(f"Successfully loaded {len(papers)} papers")
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = np.array(embedding_model.encode(list(papers.values()), show_progress_bar=True))
    
    logger.info("Initializing topic modeling")
    umap_model = UMAP(n_components=2, n_neighbors=2, min_dist=0.1, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1, prediction_data=True)
    topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model)
    
    topics, _ = topic_model.fit_transform(list(papers.values()), embeddings)
    topic_to_papers = {t: [] for t in set(topics)}
    for i, topic in enumerate(topics):
        topic_to_papers[topic].append((list(papers.keys())[i], list(papers.values())[i]))
    
    model_name = "t5-small"  # Faster than bart-large-cnn
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1, batch_size=8)
    
    all_papers_to_process = [(name, text, summarizer) for topic, papers in topic_to_papers.items() for name, text in papers]
    num_workers = max(1, os.cpu_count() - 1)
    logger.info(f"Using {num_workers} worker processes for summarization")
    
    paper_results = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_paper_summary, all_papers_to_process), total=len(all_papers_to_process), desc="Summarizing papers"))
    
    for result in results:
        if result:
            paper_results[result[0]] = result[1]
    
    output_file = "summaries.txt"
    logger.info(f"Writing summaries to {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for topic, papers_info in topic_to_papers.items():
            f.write(f"--- Topic {topic}: {topic_model.get_topic(topic)} ---\n\n")
            for paper_name, _ in papers_info:
                if paper_name in paper_results:
                    f.write(f"Paper: {paper_name}\nSummary:\n{paper_results[paper_name]}\n\n")
            f.write("="*80 + "\n\n")
    
    elapsed_time = time.time() - start_time
    logger.info(f"All summaries saved to: {output_file}")
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
