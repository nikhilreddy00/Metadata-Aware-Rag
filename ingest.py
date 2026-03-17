import os
import glob
import logging
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

DATA_DIR = "data"
COLLECTION_NAME = "dynamic_sec_filings"
QDRANT_PATH = "./qdrant_db"

def extract_metadata_from_html(file_path):
    """
    Extract basic metadata from the SEC filing HTML.
    Attempts to find the Year and Company dynamically from title or filename.
    """
    metadata = {"source": file_path}
    filename = os.path.basename(file_path).lower()
    
    # Simple static heuristic from filename if it looks like 'aapl-20220924.html'
    if "aapl" in filename or "apple" in filename:
        metadata["company"] = "Apple Inc."
    elif "tsla" in filename or "tesla" in filename:
        metadata["company"] = "Tesla Inc."
    elif "msft" in filename or "microsoft" in filename:
        metadata["company"] = "Microsoft Corp."
    else:
        # Fallback to general split
        metadata["company"] = filename.split('-')[0].upper()
    
    # Extract year using regex on the filename (e.g. 2022)
    year_match = re.search(r'(20\d{2})', filename)
    if year_match:
        metadata["year"] = int(year_match.group(1))
    else:
        metadata["year"] = 2023 # Default fallback

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            soup = BeautifulSoup(f, 'lxml')
            
            title = soup.title.string if soup.title else ""
            metadata["title"] = title.strip() if title else filename
            
            text_content = soup.get_text()[:3000].upper()
            if "10-K" in text_content:
                metadata["form_type"] = "10-K"
            elif "10-Q" in text_content:
                metadata["form_type"] = "10-Q"
            elif "8-K" in text_content:
                metadata["form_type"] = "8-K"
            else:
                metadata["form_type"] = "Unknown"
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        metadata["title"] = filename
        metadata["form_type"] = "Unknown"
        
    return metadata

def process_and_ingest():
    html_files = glob.glob(os.path.join(DATA_DIR, "*.html"))
    if not html_files:
        logger.warning(f"No HTML files found in {DATA_DIR}.")
        return

    logger.info(f"Processing {len(html_files)} HTML files for multi-dataset RAG...")

    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
    ]
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )

    all_splits = []

    for file_path in html_files:
        logger.info(f"Reading {file_path}...")
        file_metadata = extract_metadata_from_html(file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8", errors='replace') as f:
                html_content = f.read()
                
            header_splits = html_splitter.split_text(html_content)
            
            for split in header_splits:
                # Merge html headers with extracted file metadata
                split.metadata.update(file_metadata)
                
            # Filter out severely empty chunks before splitting
            header_splits = [s for s in header_splits if len(s.page_content.strip()) > 10]
                
            char_splits = text_splitter.split_documents(header_splits)
            all_splits.extend(char_splits)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"Total chunks created: {len(all_splits)}")
    if len(all_splits) == 0:
        logger.error("No valid text chunks were generated. Aborting ingestion.")
        return

    # Embed and ingest into Qdrant using the robust from_documents method
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    logger.info("Ingesting chunks into Qdrant... (doing this entirely locally for free)")
    
    QdrantVectorStore.from_documents(
        documents=all_splits,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        force_recreate=True
    )
    
    logger.info("Ingestion complete! Data is now stored in ./qdrant_db")

if __name__ == "__main__":
    process_and_ingest()

