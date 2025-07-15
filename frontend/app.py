import chainlit as cl
import os
import re
import aiohttp
import asyncio
import zipfile
import io
import csv
import time
import requests
import logging
import chardet
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime
import glob
from together import AsyncTogether
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.together import TogetherProvider
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command='deno',
    args=[
        'run',
        '-N',
        '-R=node_modules',
        '-W=node_modules',
        '--node-modules-dir=auto',
        'jsr:@pydantic/mcp-run-python',
        'stdio',
    ],
)



# Load environment variables from a .env file
load_dotenv()


class Table(BaseModel):
    """Model representing a table with metadata."""
    title: str
    description: str
    path: str
    schema: List[str]
class SelectedTables(BaseModel):
    """Model representing a collection of tables."""
    tables: List[Table]
   
class IrrilevantTables(BaseModel):
    """Model representing a collection of irrelevant tables."""
    error: bool = True
    message: str = "All tables are irrelevant."


# ==============================================================================
# 1. CLIENTS AND SETTINGS
# ==============================================================================

# Initialize the Together client for the language model
client = AsyncTogether()

model = OpenAIModel(
    'Qwen/Qwen3-235B-A22B-fp8-tput',  # model library available at https://www.together.ai/models
    provider=TogetherProvider(),
)

# Settings for the language model
settings = {
    "model": "Qwen/Qwen3-235B-A22B-fp8-tput",
    "temperature": 0.7,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "max_tokens": 8000,
}

# Dictionary of data portal APIs
dict_portal = {
    "EU": "https://data.europa.eu/api/hub/search/ckan/package_search?q=",
    "USA": "https://catalog.data.gov/api/3/action/package_search?q=",
    "CAN": "https://open.canada.ca/data/api/3/action/package_search?q=",
    "UK": "https://data.gov.uk/api/action/package_search?q=",
}

# ==============================================================================
# 2. CKAN PROCESSING TOOL LOGIC
# ==============================================================================

DOWNLOAD_FOLDER = "downloads"

# Configure logging to a file
LOG_DIR = "logs/agents"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"process_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE)]
)
logger = logging.getLogger(__name__)


def convert_json_tables_to_markdown_table(tables: List[Table]) -> str:
    """Converts a list of Table objects to a Markdown table format."""
    if not tables:
        return "No tables available."
    
    headers = ["Title", "Description", "Path", "Schema"]
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "|---" * len(headers) + "|\n"
    
    for table in tables:
        schema_str = ", ".join(table.schema)
        markdown_table += f"| {table.title} | {table.description} | {table.path} | {schema_str} |\n"
    
    return markdown_table

def clear_downloads():
    """Clears the downloads folder to ensure a fresh start."""
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)
    for f in os.listdir(DOWNLOAD_FOLDER):
        os.remove(os.path.join(DOWNLOAD_FOLDER, f))
    logger.info("🧹 Downloads folder cleared.")

async def fetch_ckan_results(ckan_query_url: str):
    """Fetches dataset results from a CKAN API endpoint."""
    match = re.search(r'limit=(\d+)', ckan_query_url)
    limit_value = int(match.group(1)) if match else 10
    ckan_query_url = re.sub(r';.*', '', ckan_query_url)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(ckan_query_url) as response:
            response.raise_for_status()
            data = await response.json()
            results = data["result"]["results"]
            
            csv_resources = [
                (r["title"], res.get("description", "No description"), res["url"])
                for r in results[:limit_value]
                for res in r.get("resources", [])
                if res.get("format", "").strip().lower() in ["csv", "zip"]
                and "-fra" not in res.get("url", "").lower()
            ]
            logger.info(f"✅ Found {len(csv_resources)} CSV/ZIP resources.")
            return csv_resources[:limit_value]

def extract_csv_from_zip(zip_url):
    """Downloads a ZIP and extracts the primary CSV file."""
    try:
        logger.info(f"⬇️ Downloading ZIP: {zip_url}")
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_file = next((name for name in z.namelist() if name.lower().endswith(".csv") and "_MetaData" not in name), None)
            if not csv_file:
                logger.warning(f"⚠️ No CSV found in ZIP: {zip_url}")
                return None
            
            save_path = os.path.join(DOWNLOAD_FOLDER, os.path.basename(csv_file))
            with z.open(csv_file) as src, open(save_path, "wb") as f:
                f.write(src.read())
            logger.info(f"✅ Extracted '{csv_file}' to '{save_path}'")
            # absolute path for consistency
            save_path = os.path.abspath(save_path)
            return save_path
    except Exception as e:
        logger.error(f"❌ Failed to extract from ZIP {zip_url}: {e}")
        return None

async def process_datasets(ckan_query_url: str):
    """Fetches dataset metadata and downloads the actual data files."""
    try:
        clear_downloads()
        resources = await fetch_ckan_results(ckan_query_url)
        
        download_tasks = []
        for title, desc, url in resources:
            if url.endswith(".zip"):
                # Run synchronous extraction in a separate thread
                task = asyncio.to_thread(extract_csv_from_zip, url)
                download_tasks.append(task)
            elif url.endswith(".csv"):
                # Handle direct CSV downloads
                pass # Not implemented for brevity, focusing on ZIPs

        downloaded_paths = await asyncio.gather(*download_tasks)
        
        results = [
            {"title": res[0], "description": res[1], "path": path, }
            for res, path in zip(resources, downloaded_paths) if path
        ]
        
        return {"result": results, "error": None}
    except Exception as e:
        logger.error(f"❌ Error in process_datasets: {e}")
        return {"result": [], "error": str(e)}

# ==============================================================================
# 3. CHAINLIT APPLICATION LOGIC
# ==============================================================================

# Read the system prompt from a markdown file
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "..", "backend", "prompts", "QueryAnalyzer.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        QUERY_ANALYZER_PROMPT = f.read()
        
    # read the coder critic system prompt
    coder_prompt_path = os.path.join(script_dir, "..", "backend", "prompts", "CoderCritic.md")
    with open(coder_prompt_path, "r", encoding="utf-8") as f:
        CODER_CRITIC_PROMPT = f.read()
except FileNotFoundError:
    QUERY_ANALYZER_PROMPT = "You are a helpful assistant. The prompt file was not found."
    CODER_CRITIC_PROMPT = "You are a helpful assistant. The prompt file was not found."

@cl.on_chat_start
def start_chat():
    """Initializes the chat session."""
    selected_portal_key = "CAN"
    top_k_results = 10
    portal_url = dict_portal.get(selected_portal_key)
    api_base_url = portal_url[: portal_url.index("api") + 3]
    formatted_prompt = QUERY_ANALYZER_PROMPT.format(
        api=api_base_url, portal=portal_url, top_k_results=top_k_results
    )
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": formatted_prompt}],
    )

@cl.step(type="tool", name="Process CKAN Datasets")
async def ckan_tool(url: str, question: str):
    """A tool step that calls the CKAN processing logic."""
    await cl.sleep(1) # Small delay for UX
    cl.context.current_step.input = url
    
    result_data = await process_datasets(url)
    
    if result_data["error"]:
        return f"An error occurred: {result_data['error']}"
        
    if not result_data["result"]:
        return "No datasets were successfully processed."
    
    print(result_data)

    # Format the results for display
    # formatted_result = "### Processed Datasets:\n"
    # for item in result_data["result"]:
    #     formatted_result += f"- **{item['title']}**\n"
    #     formatted_result += f"  - *Description*: {item['description']}\n"
    #     formatted_result += f"  - *Downloaded Path*: `{item['path']}`\n"
        
    # cl.context.current_step.output = formatted_result
    
    
    selector = Agent(
        name="SelectorAgent",
        model=model,
        system_prompt="You are an agent that approves the gathered tables, if all the tables are irrelevant, you should return error otherwise you should analyze the schema of each table. Based only on the name, description and schema determine if one or multiple tables (max 2) that are needed to answer the question. Selected only the tables that can be used to answer the question, questions require 2 tables to be selected for a join.",
        output_type=[SelectedTables, IrrilevantTables],
    )
    
    
    prompt = f"<tables>\n{result_data['result']}\n</tables> \n<question>{question}</question>"

    # change the path of the tables to absolute path

    result_data["tables"] = convert_json_tables_to_markdown_table(selector.run_sync(prompt).output.tables)
    print(f"Selected tables: {result_data['tables']}")

    return result_data

@cl.step(name="Code Execution", show_input=False)
async def code_execution(selected_tables, question):
    """
    Executes the code provided in the messages and returns the result.
    This is a placeholder for actual code execution logic.
    """
    current_step = cl.context.current_step
    current_step.output = "Executing code..."
    prompt = CODER_CRITIC_PROMPT.format(selected_tables=selected_tables, question=question)
    
    from agno.agent import Agent
    from agno.tools.python import PythonTools
    from agno.models.together import Together
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    coder = Agent(
        model=Together(id="Qwen/Qwen3-235B-A22B-fp8-tput"),
        markdown=True,
        tools=[PythonTools()],
        show_tool_calls=True,
    )
    
    code = await coder.arun(prompt)
    print(f"Generated code: {code}")
    code = code.content
    print(f"Raw generated code: {code}")
    # isolate the generated code from the rest of the response
    code = re.search(r"<python>(.*?)</python>", code, re.DOTALL).group(0) if code else ""   

    # remove the tags
    code = re.sub(r"<python>|</python>", "", code).strip()
        
    print(f"Isolated code: {code}")
    if not code:
        current_step.output = "No code generated."
        return "No code generated."
    print(f"Generated code: {code}")
    # current_step.output = f"Generated code:\n```python\n{code}\n```"
    import tempfile
    from autogen_core import CancellationToken
    from autogen_core.code_executor import with_requirements, CodeBlock
    from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
    
    result = ""
    with tempfile.TemporaryDirectory() as temp_dir:
        import venv
        venv_dir = "/home/angelo.mozzillo/lakegen_demo/.venv"
        venv_builder = venv.EnvBuilder(with_pip=True)
        venv_context = venv_builder.ensure_directories(venv_dir)
        executor = LocalCommandLineCodeExecutor(work_dir=temp_dir, virtual_env_context=venv_context)
        code_block = CodeBlock(code=code, language="python")
        
        # Execute the code block
        try:
            result = await executor.execute_code_blocks([code_block], cancellation_token=CancellationToken())
            print(f"Execution result: {result}")
            current_step.output = f"Generated code:\n```python\n{code}\n```\n\nExecution Result:\n{result}"
        except Exception as e:
            print(f"Error during code execution: {e}")
            current_step.output = f"An error occurred during code execution: {e}"


    # async with stdio_client(server_params) as (read, write):
    #     async with ClientSession(read, write) as session:
    #         await session.initialize()
    #         tools = await session.list_tools()
    #         print(len(tools.tools))
    #         #> 1
    #         print(repr(tools.tools[0].name))
    #         #> 'run_python_code'
    #         print(repr(tools.tools[0].inputSchema))
    #         """
    #         {'type': 'object', 'properties': {'python_code': {'type': 'string', 'description': 'Python code to run'}}, 'required': ['python_code'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}
    #         """
    #         result = await session.call_tool('run_python_code', {'python_code': code})
    #         print(result.content[0].text)
    #         current_step.output = f"Generated code:\n```python\n{code}\n```\n\nExecution Result:\n{result.content[0].text}"

    return f"Generated code:\n```python\n{code}\n```\n\nExecution Result:\n{result}"
    
    
    
    


@cl.step(name="Extended Thinking", show_input=False)
async def extended_thinking(messages, settings):
    """
    Sends messages to the model and processes the streaming response.
    It separates the model's "thinking" process (enclosed in <think> tags)
    from the final response.
    """
    current_step = cl.context.current_step
    current_step.output = ""

    stream = await client.chat.completions.create(
        messages=messages,
        stream=True,
        **settings,
    )

    thinking_content = ""
    final_content = ""
    in_thinking_block = False
    buffer = ""
    has_thinking = False

    async for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content or ""
        if not token:
            continue
            
        for char in token:
            buffer += char
            
            if not in_thinking_block and buffer.endswith("<think>"):
                content_before_tag = buffer[:-7]
                final_content += content_before_tag
                buffer = ""
                in_thinking_block = True
                has_thinking = True
            elif in_thinking_block and buffer.endswith("</think>"):
                content_before_tag = buffer[:-8]
                thinking_content += content_before_tag
                await current_step.stream_token(content_before_tag)
                buffer = ""
                in_thinking_block = False
            elif char in [' ', '\n', '.', ',', '!', '?', ';', ':', ')'] and buffer:
                if in_thinking_block:
                    thinking_content += buffer
                    await current_step.stream_token(buffer)
                else:
                    final_content += buffer
                buffer = ""
    
    if buffer:
        if in_thinking_block:
            thinking_content += buffer
            await current_step.stream_token(buffer)
        else:
            final_content += buffer
    
    return has_thinking, final_content

@cl.on_message
async def on_message(msg: cl.Message):
    """Handles incoming user messages, runs the model, and calls tools."""
    message_history = cl.user_session.get("message_history")
    question = msg.content.strip()
    message_history.append({"role": "user", "content": msg.content})

    # Get the model's response, which may include a tool call
    has_thinking, final_content = await extended_thinking(message_history, settings)

    # Stream the model's raw response to the user
    final_message = cl.Message(content="")
    await final_message.send()
    if final_content:
        for char in final_content:
            await final_message.stream_token(char)
    await final_message.update()

    # Add model's response to history
    if final_content.strip():
        message_history.append({"role": "assistant", "content": final_content.strip()})
        cl.user_session.set("message_history", message_history)

    # Check for a tool call in the model's response
    match = re.search(r"<ckan_url>(.*?)</ckan_url>", final_content, re.DOTALL)
    if match:
        url = match.group(1).strip()
        
        # Call the CKAN processing tool
        tool_res = await ckan_tool(url, question)

        selected_tables = tool_res.get("tables", [])

        # Send the final answer from the tool
        if tool_res and tool_res.get("tables"):
            formatted_result = f"### Successfully Processed Datasets:\n{tool_res['tables']}\n"
            await cl.Message(content=formatted_result).send()

        elif tool_res and tool_res.get("error"):
            await cl.Message(content=f"An error occurred: {tool_res['error']}").send()
        else:
            await cl.Message(content="The tool ran, but did not return any processable datasets.").send()
            
    else:
        # If no tool call, just send the final content
        await cl.Message(content=final_content).send()
    
    # generate code based on the selected tables and question
    code = await code_execution(selected_tables, question)
    
    # Send the generated code to the user
    await cl.Message(content=code).send()
