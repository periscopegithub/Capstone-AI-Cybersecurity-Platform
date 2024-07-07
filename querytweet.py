import time
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os
import pandas as pd
from langchain.schema import HumanMessage, SystemMessage
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from datetime import datetime, timezone
from langchain_community.llms import Ollama

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

# Initialize Azure LLM (GPT-4) for more complex agentic tasks
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
    api_key=os.environ.get("AZURE_OPENAI_KEY_GPT4O"),
    openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_GPT4O"),
    temperature=0,
)

# Define the columns
columns = [
    "Alert Type",
    "Ransomware",
    "Victimized Entity",
    "Hashtagged country",
    "Entity in image",
    "Country",
    "Industry",
]

username = "dexpose_io"
csv_agent = create_csv_agent(
    azure_gpt4,
    f"{username}.csv",
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

prompt = f"""You are given a dataset of historic cyber security alerts.
        'Date (UTC)' column is the alert publish date.
        'URL' column is the url to the alert.
        'Image' column is the url to the alert image.	
        'Text' column is the text content of the alert.	
        '{columns[0]}' column shows the type of alert given by the alert.
        '{columns[1]}' column shows the name of ransomware mentioned by the alert.	
        '{columns[2]}' gives the name of the victimized organization mentioned in the alert.
        '{columns[5]}' gives the country in which the victimized organization is based.
        '{columns[6]}' gives the industry of the victimized organization.
        Answer the question by appropriately analyzing the data, but ignoring columns '{columns[3]}' and '{columns[4]}' and 'UNK' values. """


def country_stats(country):
    queries = [
        f"How many times has {country} been subject to Ransomware attacks in the past 90 days?",
        f"How many times has {country} been subject to Data Breach incidents in the past 90 days?",
    ]

    responses = []
    for query in queries:
        time.sleep(15)
        response = csv_agent.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=query),
            ]
        )
        print(response)
        responses.append(response["output"])

    input_text = """
    Given the below information, write a brief summary on the cyber threat situation in his country.
    Use a formal tone. 
    Start with "According to our monitoring of the cyber threat alerts published on @dexpose_io (X.com)"
    
    """
    input_text += " ".join(responses)
    backstory = f"You are a cyber security expert who gives advice to your client. Your client's firm is based in {country}."

    output = llama3.invoke(
        [
            SystemMessage(content=backstory),
            HumanMessage(content=input_text),
        ]
    )
    print(output)
    return output


def industry_stats(industry):
    queries = [
        f"How many times has {industry} been subject to Ransomware attacks in the past 90 days?",
        f"How many times has {industry} been subject to Data Breach incidents in the past 90 days?",
    ]

    responses = []
    for query in queries:
        time.sleep(15)
        response = csv_agent.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=query),
            ]
        )
        print(response)
        responses.append(response["output"])

    input_text = """Given the below information, write a brief summary on the cyber threat situation in his industry.
    Use a formal tone. 
    Start with "According to our monitoring of the cyber threat alerts published on @dexpose_io (X.com)."
    
    """
    input_text += " ".join(responses)
    backstory = f"You are a cyber security expert who gives advice to your client. Your client's firm is in the {industry} industry."

    output = llama3.invoke(
        [
            SystemMessage(content=backstory),
            HumanMessage(content=input_text),
        ]
    )
    print(output)
    # return output.content
    return output
