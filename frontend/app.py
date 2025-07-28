import tempfile
import chainlit as cl
from chainlit.input_widget import *
import os
import re
import pandas as pd
from pydantic import BaseModel
from utils import *
from ckan import CanadaCKAN
from vector_db import MilvusDB
from autogen_core.models import UserMessage, AssistantMessage
from autogen_core.tools import FunctionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import TaskResult
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

# defining integration tools 
from tools import search_single_joins, search_unions

single_joins_tool = FunctionTool(search_single_joins, description=search_single_joins.__doc__, name="search_single_joins", strict=True)
unions_tool = FunctionTool(search_unions, description=search_unions.__doc__, name="search_unions")


class Table(BaseModel):
    """Model representing a table with metadata."""

    id: str
    name: str
    schema: List[str]


class SelectedTables(BaseModel):
    """Model representing a collection of tables."""

    selected_tables: List[Table]
    reasoning: str
    join_rationale: str


# == Tools Definition ===
@cl.step(name="Retrieval", show_input=False)
async def keyword_extraction(messages, settings):
    """
    Sends messages to the model and processes the streaming response.
    It separates the model's "thinking" process (enclosed in <think> tags)
    from the final response.
    """
    current_step = cl.context.current_step
    current_step.output = ""

    # stream = await client.chat.completions.create(
    #     messages=messages,
    #     stream=True,
    #     **settings,
    # )
    question = messages

    # print(f"Messages: {[UserMessage(source=msg['role'], content=msg['content']) for msg in messages]}")
    model_client = cl.user_session.get("model_client")
    system_prompt = get_analyzer_prompt(
        api_base_url=settings.get("model_api"),
        portal_url=settings.get("portal"),
        top_k_results=settings.get("top_k_results"),
    )
    # print(f"System prompt: {system_prompt}")  # Debugging line
    analyzer = AssistantAgent(
        name="Analyzer",
        model_client=model_client,
        system_message=system_prompt,
        model_client_stream=True,
    )
    # print(f"Messages: {messages}")

    final_content = ""
    stream = analyzer.run_stream(task=question)
    async for chunk in stream:
        if type(chunk) is TextMessage:
            final_content = chunk.content
            continue
        if type(chunk) is TaskResult:
            continue
        # current_step.output += chunk.content
        await current_step.stream_token(chunk.content)

    current_step.output += "\n"

    ckan_client = CanadaCKAN()
    vector_db = MilvusDB()
    keywords_match = re.search(r"<keywords>(.*?)</keywords>", final_content, re.DOTALL)
    keywords = keywords_match.group(1).split(",") if keywords_match else []

    ckan_response = ckan_client.package_search(
        q=keywords[0],
        rows=int(settings.get("top_k_results")),
        defType="edismax",
        sort="sort desc",
    )
    cl.user_session.set("keywords", keywords)

    packages = ckan_response["result"]["results"]
    # packages_string = [f"{p['title']} {p['notes']}" for p in packages]

    vector_db.create_collection(overwrite=True)
    vector_db.insert_documents(packages)
    result = vector_db.hybrid_search(
        query=question, limit=int(settings.get("top_k_results"))
    )

    selected_packages = [p for p in packages if p["id"] in [r["id"] for r in result]]
    for package in result:
        if package.get("id") not in [p["id"] for p in packages]:
            continue
        package_info = next((p for p in packages if p["id"] == package["id"]), None)
        if package_info:
            package_title = package_info.get("title", "No title available")
            package_notes = package_info.get("notes", "No notes available")
            package_url = f"https://open.canada.ca/data/en/dataset/{package_info['id']}"

            await current_step.stream_token(
                f"#### [{package_title}]({package_url})\n\n{package_notes}\n\n"
            )
            # current_step.output += f"#### [{package_title}]({package_url})\n\n{package_notes}\n\n"
        else:
            await current_step.stream_token(
                f"#### {package['id']}\n\nNo additional information available.\n\n"
            )

    downloaded, count = ckan_client.download_tables_from_selected_packages(
        selected_packages,
        download_path=DOWNLOAD_FOLDER,
        download_format="csv",
        max_workers=10,
        verbose=True,
    )

    tables = []
    for table in downloaded:
        # read the schema of the table
        file_path = os.path.join(table[3], f"{table[0]}.{table[4]}")
        if os.path.exists(file_path):
            if table[4] == "csv":
                df = pd.read_csv(file_path, nrows=0)
                schema = df.columns.tolist()
            elif table[4] == "parquet":
                df = pd.read_parquet(file_path, engine="pyarrow", nrows=0)
                schema = df.columns.tolist()
            else:
                schema = []
        else:
            schema = []

        tables.append(
            {
                "id": table[0],
                "name": table[1],
                "url": table[2],
                "download_path": table[3],
                "download_format": table[4],
                "schema": schema,
            }
        )

    current_step.output += (
        f"#### Downloaded {len(downloaded)} tables from the selected packages.\n\n"
    )

    return tables


@cl.step(name="Integration", show_input=False)
async def integration(retrieved_tables, question, settings):
    """
    Integrates the retrieved tables into the response.
    This function is a placeholder for future integration logic.
    """
    current_step = cl.context.current_step
    current_step.output = ""
    model_client = cl.user_session.get("model_client")
    print('\n\n\n\n', model_client, type(model_client), '\n\n\n\n\n')
    # import blend

    # indexer = blend.BLEND('blend.db')
    # _, dbcon = indexer.create_index(DOWNLOAD_FOLDER, max_workers=6, limit_table_rows=1000, verbose=True)

    # print(dbcon.table('AllTables').show())
    # print(dbcon.sql("SELECT COUNT(*) FROM AllTables;"))

    selector = AssistantAgent(
        "SelectorAgent",
        model_client=model_client,
        system_message=TABLE_SELECTOR_PROMPT,
        output_content_type=SelectedTables,
        # TODO: add tool for join discovery @nanni00
        tools=[
            single_joins_tool# , unions_tool
        ],
        # reflect_on_tool_use=False,
        reflect_on_tool_use=True,
        model_client_stream=True,
    )

    # Here you can implement the logic to integrate the retrieved tables
    # into the response. For now, we just return the retrieved tables.

    stream = selector.run_stream(
        task=f"Question: {question}\n\nTables:\n{retrieved_tables}\n\n"
    )

    async for chunk in stream:
        if type(chunk) is TextMessage:
            continue

        # Handle StructuredMessage containing SelectedTables
        if hasattr(chunk, "content") and isinstance(chunk.content, SelectedTables):
            # Convert structured output to string for current_step.output
            structured_content = chunk.content
            output_text = f"""#### Selected Tables: {", ".join([table.name for table in structured_content.selected_tables])}\n\n**Reasoning**: {structured_content.reasoning}\n\n**Join Rationale**: {structured_content.join_rationale}"""

            await current_step.stream_token(output_text)
            continue

        if type(chunk) is TaskResult:
            result = chunk
            break

    selected_tables = result.messages[-1].content if result.messages else None

    return selected_tables


@cl.step(name="Answer", show_input=False)
async def answer(selected_tables, question, settings):
    current_step = cl.context.current_step
    current_step.output = ""
    model_client = cl.user_session.get("model_client")

    """
    Executes the code provided in the messages and returns the result.
    This is a placeholder for actual code execution logic.
    """
    current_step = cl.context.current_step
    # current_step.output = "Executing code...\n"

    # samples three rows of the selected tables looking from path with csv
    print(f"Selected tables: {selected_tables}")

    # Create a prompt for the code execution agent
    # The prompt should be formatted according to the CoderCritic.md file
    # and should include the selected tables and the question.
    tables_with_rows = []
    for table in selected_tables.selected_tables:
        try:
            df = pd.read_csv(
                os.path.join(DOWNLOAD_FOLDER, f"{table.id}.csv"), nrows=10
            )  # Read only the first 3 rows
            sample_rows = df.to_markdown()
            tables_with_rows.append(
                {
                    "name": table.name,
                    # absolute path to the csv file
                    "path": os.path.abspath(
                        os.path.join(DOWNLOAD_FOLDER, f"{table.id}.csv")
                    ),
                    "schema": table.schema,
                    "sample_rows": sample_rows,
                }
            )
        except Exception as e:
            print(f"Error reading {table}: {e}")
            continue

    # format tables_with_rows as a string Table: title\n Description\n Path\n Schema\n Table Example: \n
    if not tables_with_rows:
        current_step.output = "No valid tables with sample rows found."
        return "No valid tables with sample rows found."

    # format the selected tables as a string
    selected_tables_str = "\n".join(
        [
            f"Table: {table['name']}\nPath: {table['path']}\nSchema: {', '.join(table['schema'])}\nTable Example: \n{table['sample_rows']}"
            for table in tables_with_rows
        ]
    )

    print(f"Selected tables with sample rows: {tables_with_rows}")

    prompt = CODE_EXECUTOR_PROMPT.format(
        selected_tables=selected_tables_str, question=question
    )

    result = ""
    with tempfile.TemporaryDirectory() as temp_dir:
        import venv

        try:
            venv_dir = os.path.join(os.path.dirname(__file__), "..", ".coder-venv")
            assert os.path.exists(venv_dir)
        except FileNotFoundError:
            venv_dir = os.path.join(os.path.dirname(__file__), "..", ".venv")
            assert os.path.exists(venv_dir)

        venv_builder = venv.EnvBuilder(with_pip=True)
        venv_context = venv_builder.ensure_directories(venv_dir)
        executor = LocalCommandLineCodeExecutor(
            work_dir=temp_dir, virtual_env_context=venv_context
        )

        await executor.start()

        code_execution_agent = CodeExecutorAgent(
            "CodeExecutionAgent",
            code_executor=executor,
            # model_client=model_client,
            # system_message="/nothink Only if the code execution was successful return the result of the code and write STOP\n <Result>\n STOP. Otherwise return only the error message, nothing else.",
        )

        coder = AssistantAgent(
            "CriticCoder",
            model_client=model_client,
            system_message="You are an expert Coder Critic specializing in evaluating dataset relevance and generating executable pandas code for data analysis tasks. Your primary responsibility is to assess whether provided tables can answer specific questions and generate robust Python solutions. Output 'STOP' after the Python script runs successful execution",
        )

        # termination_condition = MaxMessageTermination(6)
        text_termination = TextMentionTermination("STOP")
        # combined_termination = termination_condition | text_termination

        groupchat = RoundRobinGroupChat(
            participants=[coder, code_execution_agent],
            termination_condition=text_termination,
            max_turns=6,
        )

        result = groupchat.run_stream(task=prompt)
        final = ""
        async for chunk in result:
            if hasattr(chunk, "source") and chunk.source == "user":
                continue
            if hasattr(chunk, "messages") and chunk.messages:
                chunk = chunk.messages[-1].content
            if hasattr(chunk, "content"):
                chunk = chunk.content
            final += chunk
            await current_step.stream_token(chunk)

        await executor.stop()

    answerer = AssistantAgent(
        "Answerer",
        model_client=model_client,
        system_message=f"/nothink \nYou are an assistant that provide the final answer to the user question: <question>{question}</question> on the code execution result. If the code execution was successful, return the result of the code execution in the answer to the question with the explaination on how the result is reached. If the code execution failed, return only the code in a python code block and the error message. ",
        model_client_stream=True,
    )

    # print(final)

    #     # Run the answerer agent to get the final answer, append the name of the agent to the messages
    # messages = ''.join([msg.source + ": " + msg.content + "\n" for msg in result.messages[:-2]])
    msg = cl.Message(content="")
    final_answer = answerer.run_stream(task=final)
    async for chunk in final_answer:
        if hasattr(chunk, "source") and chunk.source == "user":
            continue
        if hasattr(chunk, "messages") and chunk.messages:
            chunk = chunk.messages[-1].content
        if hasattr(chunk, "content"):
            chunk = chunk.content
        await msg.stream_token(chunk)


# == Chainlit Settings ===


@cl.set_starters
async def starters():
    return [
        cl.Starter(
            label="Amount paid by department",
            message="What is the total amount paid by each department for supplier payments and grants in the fiscal year ended March 31, 2020, as reported in the Government of New Brunswick’s public accounts?",
        ),

        cl.Starter(
            # on keyword dataset, is the one on index 1207
            label="Union Example: Daily intake of dairy product changes",
            message="How has the daily intake of dairy products changed between 2004 and 2015 for different age groups and genders according to the Canadian Community Health Survey conducted by Health Canada? Please include details on various types of dairy products like milk, cheese, and ice cream.",
        ),

        cl.Starter(
            # on keyword dataset, is the one on index 1158
            label="Join Example: Nova Scotia licenses",
            message="Which counties in Nova Scotia have the same vendors selling both fishing and hunting licenses, and how many addresses do these vendors have? Please provide the number of addresses for each vendor that appears in both lists.",
        )
    ]


@cl.on_chat_start
async def start_chat():
    """Initializes the chat session."""
    settings = await cl.ChatSettings(
        [
            Select(
                id="model_api",
                label="Models API URL",
                values=[
                    "https://api.together.xyz/v1/",
                    "https://api.openai.com/v1/",
                    "https://api.mistral.ai/v1/",
                ],
                initial_index=0,
            ),
            Switch(id="talk", label="Talk with the Agent", initial=False),
            Select(
                id="portal",
                label="Open Data Portal",
                values=["EU", "USA", "CAN", "UK"],
                initial_index=2,
            ),
            NumberInput(
                id="top_k_results",
                label="Top K Results",
                initial=10,
                step=1,
            ),
        ]
    ).send()

    cl.user_session.set("settings", settings)

    cl.user_session.set("model_client", get_model_client(settings))


@cl.on_settings_update
async def setup_api_key(settings):
    cl.user_session.set("settings", settings)


@cl.on_message
async def on_message(msg: cl.Message):
    # Print selected settings
    # TODO: Use action to implement the talk to agent mode
    settings = cl.user_session.get("settings")
    portal_key = settings.get("portal")
    top_k_results = settings.get("top_k_results")

    question = msg.content
    cl.user_session.set("question", question)
    tables = await keyword_extraction(messages=msg.content, settings=settings)

    selected_tables = await integration(
        retrieved_tables=tables, question=question, settings=settings
    )
    await answer(selected_tables=selected_tables, question=question, settings=settings)
