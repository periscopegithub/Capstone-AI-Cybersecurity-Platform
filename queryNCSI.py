from crewai_tools import ScrapeWebsiteTool
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.llms import Ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Load API keys and configurations from .env file as environment variables
load_dotenv()

# Initialize Llama3 run locally
llama3 = Ollama(model="llama3")

# Initialize Azure LLM (GPT-3.5 Turbo) for simple text processing
azure_gpt35turbo = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_GPT35TURBO"),
    temperature=0,
)

# Initialize Azure LLM (GPT-4) for more more complex agentic tasks
azure_gpt4 = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_GPT4"),
    temperature=0,
)

# Initialize Azure LLM (GPT-4o) for more more complex agentic tasks
azure_gpt4o = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT_GPT4O"),
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_GPT4O"),
    temperature=0,
)

# Initialize the tool with the website URL, so the agent can only scrap the content of the specified website
scrape_ranking = ScrapeWebsiteTool(website_url="https://ncsi.ega.ee/ncsi-index/")
scrape_methodology = ScrapeWebsiteTool(
    website_url="https://r.jina.ai/https://ncsi.ega.ee/methodology/"
)

ncsi_ranking = scrape_ranking.run()
ncsi_methodology = scrape_methodology.run()


def ncsi_insights(country):

    ncsi = ncsi_methodology + "\nNCSI Rankings:\n" + ncsi_ranking

    prompt = f"""You are given scraped content from the National Cyber Security Index (NCSI) website.
        The webpage presents the index methodology and a table containing data: Rank, Country, National Cyber Security Index, Digital Development Level, Difference
        'Rank' refers to the country's rank in the index.
        'National Cyber Security Index' refers to the country's index score.
        'Digital Development Level' refers to the country's score in Digital Development Level.
        'Difference' is simply the difference between the scores in 'National Cyber Security Index' and 'Digital Development Level'.
        Answer the question by referring to the scraped content as follows: \n {ncsi}"""

    queries = [
        "Provide a brief introduction to the NSCI index.",
        "How many country rankings are there in the NSCI index?",
        f"What is the rank of {country}?",
        f"What is {country}'s National Cyber Security Index?",
    ]

    def process_query(query):
        response = llama3.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=query),
            ]
        )
        return response

    responses = []

    # for query in queries:
    #     time.sleep(10)
    #     response = azure_gpt35turbo.invoke(
    #         [
    #             SystemMessage(content=prompt),
    #             HumanMessage(content=query),
    #         ]
    #     )
    #     responses.append(response.content)

    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(process_query, query): query for query in queries
        }

        # Log the number of threads being used
        logging.info(f"Number of threads used: {executor._max_workers}")

        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                response = future.result()
                responses.append(response)
            except Exception as e:
                logging.error(f"Error processing query '{query}': {e}", exc_info=True)

    for response in responses:
        print(response)

    input_text = " ".join(responses)
    summary_prompt = f"""You are a cyber security expert and is studying the National Cyber Security Index (NCSI).
    Given the below information, write a brief (within 5 sentences) with some insights into the NCSI index and {country}'s index standing.
    Use a formal tone. 
    Start with "According to the latest National Cyber Security Index (NCSI),"."""

    output = llama3.invoke(
        [
            SystemMessage(content=summary_prompt),
            HumanMessage(content=input_text),
        ]
    )
    print(output)
    return output
