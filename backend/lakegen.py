import os
import sys
import json
import shutil
import asyncio
from functools import lru_cache
from collections import defaultdict
from typing import Any, Dict, Annotated

import pandas as pd
import pandas.api.types as ptypes
from dotenv import load_dotenv

# for BLEND
import duckdb
import bidict

from process_ckan import process_datasets
from typing_extensions import Annotated

from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent, UserProxyAgent, SocietyOfMindAgent 
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor


# set seed
import random
import numpy as np

random.seed(42)
np.random.seed(42)

os.environ["no_proxy"] = "localhost,127.0.0.1,geonext.comune.modena.it"
sys.path.append("backend")


# Create a function tool.
process_ckan_results_tool = FunctionTool(
    process_datasets,
    description="Function for executing CKAN query and returning a response",
)


@lru_cache(maxsize=int(1e3))
def clear_string(s: Any):
    """Now, no actual clearing is done."""
    return str(s)


async def table_sampling(selected_tables: list) -> Annotated[
    Dict[str, str],
    "Dictionary with keys 'result' (table details) and 'error' (if any error occurs)",
]:
    print("Entrato")
    print(selected_tables)
    
    # to define as function parameter?
    K : int = 3

    sampled_tables = []
    if type(selected_tables) is str:
        selected_tables = json.loads(selected_tables)

    # create a in-memory duckdb database
    blend_tmp_con = duckdb.connect()
    blend_tmp_con.sql(
        """
        CREATE TABLE AllTables (
            TableId     INT,
            ColumnId    INT,
            RowId       INT,
            CellValue   INT
        );
        """
    )

    # a bidirectional-values dictionary, this avoids storing
    # original values into the index, which increases memory
    # and insert/search time usage (to just get the overlap,
    # this is not necessary, only if we want to see which 
    # values overlap)
    values_bidict = defaultdict(int)

    # used to create final results
    tables_schema = {}

    for table_id, t in enumerate(selected_tables):
        try:
            print(t)
            df = pd.read_csv(t["path"], on_bad_lines="skip")
            tables_schema[t["title"]] = df.columns.to_list()

            table_sample = df.iloc[:3]

            sampled_tables.append(
                {
                    "title": t["title"],
                    "description": t["description"],
                    "table_sample": str(table_sample.to_json(orient="records")),
                    "path": t["path"],
                }
            )

            # Add values to the bidict, exluding numeric and boolean columns 
            # which can cause casual joins and are generally filtered in these operations
            for column_name in df.select_dtypes(exclude=["number", "bool"]).columns:
                for value in map(clear_string, df[column_name].dropna().unique()):
                    if value not in values_bidict:
                        values_bidict[value] = len(values_bidict)
                
            # create the records for the BLEND-like index
            table_blend_records = [
                [
                    table_id, 
                    column_id, 
                    row_id, 
                    values_bidict[clear_string(cell)] if clear_string(cell) in values_bidict else pd.NA
                ]
                for row_id, row in df.iterrows()
                for column_id, cell in enumerate(row)
            ]

            # insert the records through the duckdb API, filtering those records with a nan,
            # which are numeric and boolean values (to optimize this step...)
            records_df = pd.DataFrame(table_blend_records, columns=['TableId', 'ColumnId', 'RowId', 'CellValue']).dropna(axis=0, how='any')
            blend_tmp_con.execute(query="INSERT INTO AllTables SELECT * FROM records_df;")
            blend_tmp_con.commit()
        except Exception as e:
            print(e)

    # actually create a "bi-dictionary"
    values_bidict = bidict.bidict(values_bidict)

    # a list of tuples <q_table_title, r_table_title, q_column, r_column, overlap_size>
    final_results = []

    # for each table, query with the Single-Column Seeker approach from BLEND
    for table_id, t in enumerate(selected_tables):
        query_table = pd.read_csv(t["path"], on_bad_lines="skip")        
        for query_column_name in query_table.columns:
            query_column = query_table[query_column_name]
            # again, filter numeric and boolean columns 
            # (not done before, to avoid mismatch into column index number)
            if ptypes.is_numeric_dtype(query_column) or ptypes.is_bool_dtype(query_column):
                continue
            
            query_column = query_column.apply(lambda v: values_bidict[clear_string(v)])
            results = blend_tmp_con.sql(f"""
                SELECT TableId, ColumnId, COUNT(DISTINCT CellValue) AS overlap 
                FROM AllTables
                WHERE CellValue IN ({ ','.join(query_column) })
                AND TableId <> {table_id}
                GROUP BY TableId, ColumnId
                ORDER BY COUNT(DISTINCT CellValue) DESC
                LIMIT {K};
            """).fetchall()

            for r_table_id, r_column_id, overlap in results:
                final_results.append(
                    [
                        t["title"],
                        selected_tables[r_table_id]["title"],
                        query_column_name,
                        tables_schema[selected_tables[r_table_id]["title"]][r_column_id],
                        overlap
                    ]
                )


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
            "cache_seed"        : None,
            "json_output"       : False,
            "vision"            : False,
            "function_calling"  : True,
        },
        temperature=0.0,
        seed=42,
    )


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
        4. Extract MAX 2-3 keywords (separated by +, keyword1+keyword2+..., like dog+animal) to maximize dataset matches while preserving relevance, remember to use the identified language.
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
        Remember to separate keywords with +

        """,
        model_client=model_client,
    )

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

    relevant_tables_critic = AssistantAgent(
        "RelevantTablesCritic",
        model_client=model_client,
        system_message=f"""You are a critic.
            Respond with 'APPROVE' if some of the tables are relevant to the question:
            {question}. 
            
            Otherwise write only the question: 
            {question}"""
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
        participants=[
            query_analyzer, 
            ckan_generator, 
            relevant_tables_critic
        ],
        termination_condition=inner_termination,
        max_turns=6,
    )

    ckan_mind = SocietyOfMindAgent(
        name="ckan_mind",
        team=inner_team,
        model_client=model_client,
        instruction=""""
        You have only to return a dictionary of gathered tables to pass to the next agent.
        Output only this dictionary:
        {
            "tables": [
                {
                "title": "<title>",
                "description": "<desc>",
                "path": "<url>",
                "schema": <schema>
                },
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
    
    integrator = RoundRobinGroupChat(
        participants=[
            ckan_mind, 
            table_selector
        ], 
        max_turns=2,
        termination_condition=TextMentionTermination("STOP") 
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
            participants=[
                critic, 
                code_executor_agent, 
                agent4
            ],
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
        "MO"    : "https://opendata.comune.modena.it/api/3/action/package_search?q=",
        "IT"    : "https://www.dati.gov.it/opendata/api/3/action/package_search?q=",
        "USA"   : "https://catalog.data.gov/api/3/action/package_search?q=",
        "CAN"   : "https://open.canada.ca/data/api/3/action/package_search?q=",
        "UK"    : "https://data.gov.uk/api/3/action/package_search?q=",
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