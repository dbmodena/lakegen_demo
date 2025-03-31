import os

os.environ["no_proxy"] = "localhost,127.0.0.1,geonext.comune.modena.it"

import streamlit as st
from dotenv import load_dotenv
import asyncio
import autogen
import json
import requests
from typing import Dict, Annotated
import pandas as pd
import io
from contextlib import redirect_stdout, redirect_stderr
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.coding import LocalCommandLineCodeExecutor
import sqlite3
from dataclasses import dataclass
from autogen_ext.models.openai import OpenAIChatCompletionClient

import os

import shutil
import requests
import csv
import zipfile
from urllib.parse import urlparse

import sys

sys.path.append("backend")


from process_ckan import process_datasets
from autogen_core.tools import FunctionTool
from typing_extensions import Annotated


# set seed
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# Create a function tool.
process_ckan_results_tool = FunctionTool(
    process_datasets,
    description="Function for executing CKAN query and returning a response",
)


async def table_sampling(selected_tables: list) -> Annotated[
    Dict[str, str],
    "Dictionary with keys 'result' (table details) and 'error' (if any error occurs)",
]:
    print("Entrato")
    print(selected_tables)
    sampled_tables = []
    if type(selected_tables) is str:
        selected_tables = json.loads(selected_tables)

    for t in selected_tables:
        try:
            print(t)
            table_sample = pd.read_csv(t["path"], nrows=3)
            sampled_tables.append(
                {
                    "title": t["title"],
                    "description": t["description"],
                    "table_sample": str(table_sample.to_json(orient="records")),
                    "path": t["path"],
                }
            )

        except Exception as e:
            print(e)

    if not selected_tables:
        return {"result": [], "error": "empty tables, regenerate keywords"}

    print("----------------SELECTED TABLES---------------------")
    print(sampled_tables)
    if len(sampled_tables) < 2:
        return {"result": [], "error": "not enough tables, regenerate keywords"}

    # save to a file and overwrite
    with open("sampled_tables.json", "w") as f:
        json.dump(sampled_tables, f)

    return sampled_tables


# Create a function tool.
table_sampling_tool = FunctionTool(
    table_sampling,
    description="Function for sample rows of selected tables and returning a response",
)


def get_model_client(model_name):  # type: ignore
    "Mimic OpenAI API using Local LLM Server."
    return OpenAIChatCompletionClient(
        model=model_name,
        api_key="NotRequiredSinceWeAreLocal",
        base_url="http://0.0.0.0:4323",
        model_capabilities={
            "cache_seed": None,
            "json_output": False,
            "vision": False,
            "function_calling": True,
        },
        temperature=0.0,
        seed=42,
    )


import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.agents import UserProxyAgent
from autogen_ext.tools.code_execution import PythonCodeExecutionTool

load_dotenv()


async def main(question: str, portal: str, model_name: str, top_k_results: int):
    model_client = get_model_client(model_name)

    # delete sample_tables.json
    if os.path.exists("sampled_tables.json"):
        os.remove("sampled_tables.json")

    # remove coding folder
    if os.path.exists("coding"):
        shutil.rmtree("coding")

    api = portal[: portal.index("api") + 3]
    query_analyzer = AssistantAgent(
        name="QueryAnalyzer",
        system_message=f"""You are a Query Analyzer.  
        API URL : {api}
        Your job is to:
        1. Identify the main intent of the query (e.g., trends, counts, correlations).
        If the api url is in a language from a specific country (e.g., English for Canadian, Italian for italian resources), translate it into the appropriate language (removing aggregation average, sum etc).
        Identified Language:
        Translated Query in the identified language:
        2. Break it into components:
        - Subject: The main entity being queried (e.g., population, businesses).
        - Location: Geographic focus (e.g., city name), if non leave blank.
        - Filters: Time range, demographics, or other filters, if non leave blank.
        3. Simplify the query, remember to use the identified language.:
        - Generalize terms to their broader categories.
        - Remove aggregations 
        - Remove specific filters like explicit values or names.
        - Ensure the location keyword is included for accurate dataset retrieval, if non leave blank.
        4. Extract MAX 2-3 keywords (separated by +, keyword1+keyword2+...) to maximize dataset matches while preserving relevance, remember to use the identified language.
        5. Generate three query:
            1. use generalized terms, 
            2. include synonyms or alternative terms for broader coverage, 
            3. The last keyword should be the location
            4. all words should be lowercase
            5. do not use blank spaces
            
        6. Select the most general one
        7. Output one CKAN query URLs using these keywords, Always end with location:
        {portal}<keywords>;limit={top_k_results}

        Format your answer in markdown to prettify and make readable.
        """,
        model_client=model_client,
    )

    # user_proxy = AssistantAgent(name="User")

    ckan_generator = AssistantAgent(
        name="CKANGenerator",
        system_message="""
        You are a CKANGenerator. 
        Return the results from the CKAN query provided by the Query Analyzer. call process_ckan_results().
        The function returns a dictionary with the following structure:
        {
        "tables": [
            {
            "title": "<title>",
            "description": "<desc>",
            "path": "<url>",
            "schema": <schema>
            }
        ]
        }
        
        Output only the result call process_ckan_results().""",
        model_client=model_client,
        tools=[process_ckan_results_tool],
    )

    agent3 = AssistantAgent(
        "assistant3",
        model_client=model_client,
        system_message=f"You are a critic. Respond with 'APPROVE' if some of the tables are relevant to the question: {question}. Otherwise write only the question: {question}",
    )

    table_selector = AssistantAgent(
        name="TableSelector",
        system_message="""
        You are a Table Selector.  
        Always respond in English.
        Given a question and a list of tables, analyze the schema of each table.
        Based only on the name, description and schema determine if one or multiple tables (2) that are needed to answer the question.
        Selected only the tables that can be used to answer the question, questions require 2 tables to be selected for a join.
        Report the table(s) as a list of dictionaries with the attributes:
        
        selected_tables = [
            {
            "title": "<title>",
            "description": "<desc>",
            "path": "<url>"
            },
            {
            "title": "<title>",
            "description": "<desc>",
            "path": "<url>"
            }
        ]
        
        Now you have to sample the selected tables using table_sampling() and return the result.
        Say 'STOP' to stop the conversation.
        """,
        model_client=model_client,
        tools=[table_sampling_tool],
    )

    inner_termination = TextMentionTermination("APPROVE")

    inner_team = RoundRobinGroupChat(
        [query_analyzer, ckan_generator, agent3],
        termination_condition=inner_termination,
        max_turns=6,
    )

    ckan_mind = SocietyOfMindAgent(
        "ckan_mind",
        team=inner_team,
        model_client=model_client,
        instruction=""""
        You have only to return a dictionary of gatered tables to pass to the next agent.
        Output only this dictionary:
        {
        "tables": [
            {
            "title": "<title>",
            "description": "<desc>",
            "path": "<url>",
            "schema": <schema>
            }
            ...
        ]
        }
        DO NOT ADD ANY OTHER INFORMATION.
        """,
        # instruction=f"""
        # Given the question '{QUESTION}' if the results from the CKAN query are relevant to the question.
        # Otherwise, ask the Query Analyzer to generate a new query based on the original question.
        # Be very coicise in your answers.
        # """,
    )

    tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))

    termination = TextMentionTermination("STOP")

    integrator = RoundRobinGroupChat(
        [ckan_mind, table_selector], termination_condition=termination, max_turns=2
    )

    final_result = []

    # await Console(table_selection)

    async for result in integrator.run_stream(task=question):
        final_result.append(result)

    # print(final_result[-1])

    with open("sampled_tables.json", "r") as f:
        selected_tables = json.load(f)

        # tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))
        critic = AssistantAgent(
            name="Critic",
            system_message=f"""
            You are the Critic.
            
            selected_tables: {selected_tables}

            - Always maintain the English language.
            - Remember to **not** consider aggregation in the query.
            - The question is: {question}
            - The selected tables are: {selected_tables}
            - The selected tables are the only ones that can be used to answer the question.
            - If the selected tables are relevant to the question, say 'APPROVE' and generate a plan to write Python code using pandas.
            - If the selected tables are not relevant to the question, say 'STOP' to stop the conversation.

            1. **If the selected tables almost meet the criteria (e.g., partially relevant but not fully aligned)**, **do not** mention this in the answer:
                - Generate a plan to write Python code using pandas:
                    - **Does it need a join?** Define both `right_on` and `left_on`, specifying both for join.
                    - **Does it need an aggregation?** Define the aggregation (average, count, etc.). NB: The aggregation should be performed on the **resulting dataframe**.
                    - **Does it need a groupby?** Define the groupby.
                    - **Does it need a transformation?** Define the transformation.
                    - **Does it need a filter?** Define the filter.
                - Generate **Python code** inside a code block to answer the question.
                - Python should use pandas operation using the provided file path in `pd.read_csv` for the table(s).
                    - Follow the plan you generated.
                    - USE ONLY THE PATH WITH pd.read_csv(path) DO NOT CREATE DATAFRAME FROM 'table_sample' filed
                    - Make sure to use the **correct path** inside the pd.read_csv(path) statement for both tables.
                    - If a join should be performed, use `left_on` and `right_on`.
                    - Group the result if aggregation is needed and apply the aggregation on the result.
                    - Print the variables if a single value is requested in the question.
                    - **Do not use table samples** provided in the conversation to create or replace the `pd.read_csv`.
                    - **Do not invent or substitute the file path** if the correct answer cannot be found. Always use the **actual file path** provided.
                    - Remove duplicates using `drop_duplicates()` method where applicable.
                    - the result should be not empty
                     - If the result is a single value, print the value.
                     - elese if the result is a dataframe, print the first 10 rows.

                    - If you don't know how to answer, say 'I don't know'.
            2. **If the selected tables do not meet the criteria**:
                - Say **'STOP'** to stop the conversation.
            """,
            model_client=model_client,
        )

        from autogen_agentchat.agents import CodeExecutorAgent

        code_executor_agent = CodeExecutorAgent(
            "code_executor",
            code_executor=LocalCommandLineCodeExecutor(work_dir="coding"),
        )

        agent4 = AssistantAgent(
            "assistant4",
            model_client=model_client,
            system_message=f"""You are a critic. Respond with 'APPROVE' if the output to the question: {question} is in line with result. 
            Otherwise ask to regenerate the code with correction
            """,
        )

        executor_team = RoundRobinGroupChat(
            [critic, code_executor_agent, agent4],
            termination_condition=inner_termination,
            max_turns=12,
        )

        # read output

        executor_mind = SocietyOfMindAgent(
            "executor_mind",
            team=executor_team,
            model_client=model_client,
            instruction=f"""
            Write ONLY a coincise short answer to the question and if ther is a dataframe show only 10 rows in markdown and prettify and make readable, do not add information about code, only answer and dataframe.
            If the dataframe is not present do not invent an answer, write an error message.
            """,
        )

        # final_result = []
        async for result in executor_mind.run_stream(
            task=f"{question} {selected_tables}"
        ):
            final_result.append(result)

    print(final_result[-1])

    return final_result


if __name__ == "__main__":
    dict_portal = {
        "MO": "https://opendata.comune.modena.it/api/3/action/package_search?q=",
        "IT": "https://www.dati.gov.it/opendata/api/3/action/package_search?q=",
        "USA": "https://catalog.data.gov/api/3/action/package_search?q=",
        "CAN": "https://open.canada.ca/data/api/3/action/package_search?q=",
        "UK": "https://data.gov.uk/api/3/action/package_search?q=",
    }
    # get api until 'api' in the url
    portal = dict_portal["CAN"]
    question = (
        "What is the average salary of teachers with the most working days in Canada?"
    )
    asyncio.run(
        main(
            question=question,
            portal=portal,
            model_name="llama-3.3-70b-versatile",
            top_k_results=10,
        )
    )
