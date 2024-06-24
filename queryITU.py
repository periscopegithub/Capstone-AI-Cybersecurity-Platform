import pickle
import os
import time
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.llms.ollama import Ollama as llama_index_Ollama
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.llms import Ollama as langchain_Ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys and configurations from .env file as environment variables
load_dotenv()


azure_llm = AzureOpenAI(
    model=os.environ.get("AZURE_OPENAI_MODEL"),
    deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT_GPT4"),
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version=os.environ.get("AZURE_OPENAI_VERSION"),
)

Settings.llm = azure_llm


# Load data from the given file path
def load_data(filepath):
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            if not all(
                key in data for key in ["base_nodes", "objects", "recursive_index"]
            ):
                raise ValueError("Pickle file is missing one or more required keys.")
            return data["base_nodes"], data["objects"], data["recursive_index"]
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


# Generate ITU insights for a specific country
def gci_insights(country):
    base_nodes, objects, recursive_index = load_data("./GCI report/ITU-20240624.pkl")
    if not base_nodes or not objects or not recursive_index:
        logger.error("Failed to load data. Exiting...")
        return

    # pillars = "legal measures, technical measures, organizational measures, capacity development measures, and cooperative measures"

    reranker = FlagEmbeddingReranker(
        top_n=10,
        model="BAAI/bge-reranker-large",
    )

    recursive_query_engine = recursive_index.as_query_engine(
        similarity_top_k=15, node_postprocessors=[reranker], verbose=True
    )

    queries = [
        "Provide a brief introduction to Global Cybersecurity Index (GCI).",
        "Summarize the five pillars of GCI.",
        f"What is the global score of {country}?",
    ]

    def process_query(query):
        try:
            response = recursive_query_engine.query(query)
            return response.response
        except Exception as e:
            logger.error(f"Error querying '{query}': {e}")
            return None

    responses = []
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        future_to_query = {
            executor.submit(process_query, query): query for query in queries
        }
        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                result = future.result()
                if result:
                    responses.append(result)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")

    for response in responses:
        logger.info(response)

    # Initialize Llama3 running locally to summarize the responses
    lc_llama3 = langchain_Ollama(model="llama3")

    input_text = " ".join(responses)
    summary_prompt = f"""You are a cyber security expert and is studying the Global Cybersecurity Index (GCI).
    Given the below information, write a brief (within 5 sentences) with some insights into GCI index and {country}'s index standing.
    Use a formal tone. 
    Start with "According to the latest Global Cybersecurity Index (GCI),"."""

    output = lc_llama3.invoke(
        [
            SystemMessage(content=summary_prompt),
            HumanMessage(content=input_text),
        ]
    )
    print(output)
    return output


# output = gci_insights("UNITED STATES")
