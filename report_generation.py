from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os
import pandas as pd
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.llms import Ollama
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

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


def generate_detailed_response_report(questions_df, answers_df):
    category_reports = {}

    # Group questions by Category
    grouped_questions = questions_df.groupby("Question Category")

    for category, group in grouped_questions:
        # Initialize the list for this category in the dictionary
        category_reports[category] = [
            f"In the '{category}' category, the respondent answered {len(group)} questions."
        ]

        # Iterate through each question in the group
        for _, question in group.iterrows():
            question_id = question["Question ID"]
            subcategory = question["Subcategory"]
            question_text = question["Question Text"]
            score_descriptions = [question[f"Score {i}"] for i in range(1, 6)]

            # Find the respondent's answer to this question in answers_df
            respondent_answer = answers_df[answers_df["Question ID"] == question_id]
            if not respondent_answer.empty:
                respondent_score_description = respondent_answer.iloc[0][
                    "Score Description"
                ]
            else:
                respondent_score_description = "No response"

            # Prepare the score description text
            score_description_text = "', '".join(score_descriptions)

            # Append the detailed information about this question to the report list for this category
            category_reports[category].append(
                f"In respect of {subcategory}, the respondent is asked '{question_text}'. "
                f"The available answer options are '{score_description_text}'. "
                f"His response is '{respondent_score_description}'."
            )

    return category_reports


def process_category(category, response_reports, category_averages):
    logging.info(
        f"Processing category '{category}' in thread {threading.current_thread().name}"
    )

    prompt = f"""You are a cyber security expert who gives advice to your client.
    To assess the client's ('Respondent') performance in cyber security readiness, you invited him to complete a survey.
    You will review his survey answers, and write a report on his cyber security readiness in terms of '{category}'.
    Use a formal tone. Address your client as 'your organization' in the report. 
    Do not use email style, ie. do not start with "Dear Client".
    Start with the title '{category}' using markdown '###' heading format.
    """

    if category in response_reports:
        message = f"Respondent's performance for {category}:\n"
        for report in response_reports[category]:
            message += report + "\n"

        response = llama3.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=message),
            ]
        )
        return response
    return ""


def comment_each_section(
    response_reports, category_averages, overall_score_message, detailed_results
):
    # Looping through all categories in category_averages to print reports
    responses = []

    with ThreadPoolExecutor() as executor:
        future_to_category = {
            executor.submit(
                process_category, category, response_reports, category_averages
            ): category
            for category in category_averages.keys()
        }

        logging.info(f"Number of threads used: {executor._max_workers}")

        for future in as_completed(future_to_category):
            response = future.result()
            if response:
                responses.append(response)

    input_text = " ".join(responses)
    category_results = " ".join(detailed_results.values())
    survey_scores = overall_score_message + " " + category_results
    survey_scores = (
        survey_scores.replace("you", "Client")
        .replace("You", "Client")
        .replace("\n", " ")
    )
    summary_prompt = f"""You are a cyber security expert assessing the cyber security readiness of your client's organization.
    Your client has responded to a survey, and his survey score analysis is as follows:
    {survey_scores}
    Based on the above score analysis together with the following analysis of his performance by question category which he received from you, 
    write a brief overview (within 5 sentences) on his overall performance.  
    You may highlight areas of strength and areas for improvement without going into too much details.
    Use a formal tone. Address your client as 'your organization' in the report. 
    Do not use email style, ie. do not start with "Dear Client".
    Start with the title 'Overall Performance' using markdown '###' heading format.
    """

    output = llama3.invoke(
        [
            SystemMessage(content=summary_prompt),
            HumanMessage(content=input_text),
        ]
    )
    print(output)
    responses.insert(0, output)

    return responses
