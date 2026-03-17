import os
import logging
from dotenv import load_dotenv

# Ragas & Datasets
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# LangChain models
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# App Logic (we import our own retriever and rag chain)
from app import build_retriever, build_rag_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "apple_sec_filings"

# A golden dataset of typical questions we expect the Self-Querying RAG to answer
# 'ground_truth' represents the ideal answer we'd expect
GOLDEN_DATASET = [
    {
        "question": "What is Apple's primary form of revenue as reported in the 10-K?",
        "ground_truth": "Apple's primary form of revenue is the sale of its hardware products (iPhone, Mac, iPad) and related services."
    },
    {
        "question": "What are the common risk factors mentioned in Apple's filings?",
        "ground_truth": "Common risk factors include global economic conditions, severe competition, supply chain disruptions, and dependence on third-party software/component providers."
    }
]

def run_evaluation():
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY is not set.")
        return
        
    logger.info("Initializing components for Evaluation...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        client = QdrantClient(path=QDRANT_PATH)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return

    # We will use Groq's 70b model for better reasoning in the 'judge' role.
    eval_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    retriever = build_retriever(vector_store, eval_llm)
    rag_chain = build_rag_chain(retriever, eval_llm)
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    logger.info("Generating answers for the Golden Dataset...")
    for item in GOLDEN_DATASET:
        question = item["question"]
        gt = item["ground_truth"]
        
        # 1. Retrieve contexts
        retrieved_docs = retriever.invoke(question)
        doc_contents = [doc.page_content for doc in retrieved_docs]
        
        # 2. Generate answer
        answer = rag_chain.invoke(question)
        
        questions.append(question)
        answers.append(answer)
        contexts.append(doc_contents)
        ground_truths.append(gt)

    # 3. Format as HuggingFace Dataset (required by Ragas)
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(eval_data)
    
    logger.info("Running Ragas Evaluation Metrics...")
    
    # Ragas needs LLMs and Embeddings to evaluate metrics:
    metrics = [
        faithfulness,       # Hallucination check
        answer_relevancy,   # Does it actually answer the question?
        context_precision,  # Did we retrieve relevant items first?
        context_recall      # Did we retrieve ALL the necessary context?
    ]
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=embeddings
    )
    
    logger.info("=== EVALUATION RESULTS ===")
    print(result)
    
    # Save the dataframe for later analysis
    df = result.to_pandas()
    df.to_csv("ragas_evaluation_report.csv", index=False)
    logger.info("Saved detailed report to ragas_evaluation_report.csv")

if __name__ == "__main__":
    run_evaluation()
