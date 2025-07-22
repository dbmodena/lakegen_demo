import chainlit as cl
import os
import re
import aiohttp
import asyncio
import zipfile
import io
import csv
import time
from enum import Enum
from typing import Any, Dict, Annotated, Literal
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
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_core import SingleThreadedAgentRuntime
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage, CodeExecutionEvent
import tempfile
from autogen_core import CancellationToken
from autogen_core.code_executor import with_requirements, CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
# from pydantic_ai import Agent, RunContext
# from pydantic_ai.models.openai import OpenAIModel
# from pydantic_ai.providers.together import TogetherProvider
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage, AssistantMessage

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
    #relevance: Literal["relevant", "irrelevant"]
   
class IrrilevantTables(BaseModel):
    """Model representing a collection of irrelevant tables."""
    error: bool = True
    message: str = "All tables are irrelevant."


# This action callback is triggered by the callAction function in the JSX.
@cl.action_callback("on_select")
async def on_select(action: cl.Action):
    """
    Handles the selection from the dropdown.
    """
    # The payload from the frontend is available in action.payload
    selected_value = action.payload.get("value")
    
    # You can perform any backend logic here.
    # For this example, we'll just send a confirmation message.
    await cl.Message(
        content=f"You selected: **{selected_value}**"
    ).send()
    
    # You can optionally remove the action button after it's been handled
    await action.remove()
    
@cl.action_callback("on_number_change")
async def on_number_change(action: cl.Action):
    """
    Handles the new value from the number input.
    """
    new_value = action.payload.get("value")
    
    # You can perform any backend logic with the new value.
    # For this example, we just confirm the change.
    await cl.Message(content=f"Number updated to: **{new_value}**").send()
    await action.remove()

@cl.action_callback("submit_api_key")
async def on_submit_api_key(action: cl.Action):
    """
    Handles the submitted API key.
    For security, you should store it in the user_session, not in a message.
    """
    submitted_key = action.payload.get("key")
    
    # Store the key securely in the user session
    cl.user_session.set("api_key", submitted_key)
    
    # Let the user know the key was received and stored
    await cl.Message(content="Your API key has been stored securely for this session.").send()
    # The frontend will automatically update to the "submitted" view
    # because we called updateElement in the JSX.
    await action.remove()


# ==============================================================================
# 1. CLIENTS AND SETTINGS
# ==============================================================================

# Initialize the Together client for the language model
# client = AsyncTogether()

# model = OpenAIModel(
#     'Qwen/Qwen3-235B-A22B-fp8-tput',  # model library available at https://www.together.ai/models
#     provider=TogetherProvider(),
# )

# Settings for the language model
settings = {
    "model": "Qwen/Qwen3-235B-A22B-fp8-tput",
    "temperature": 0.7,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "max_tokens": 10000,
}


model_client = OpenAIChatCompletionClient(
    base_url = "https://api.together.xyz/v1/",
    api_key=os.getenv("TOGETHER_API_KEY", ""),
    model_info= {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True
    },
    **settings
    
)


# ping the model client to the global scope
try:
   result = model_client.create([UserMessage(content="Hi", source="user")])  # type: ignore
   print("Model client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize model client: {e}")

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
        
        # for each result, add the schema retrieved from the CSV file
        for result in results:
            try:
                import csv
                # read only the first row to get the schema
                with open(result["path"], "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    first_row = next(reader)
                    result["schema"] = first_row
            except Exception as e:
                logger.error(f"❌ Error reading schema for {result['title']}: {e}")
                result["schema"] = []
        
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
async def start_chat():
    """Initializes the chat session."""
    
    number_props = {
        "label": "Top K Results",
        "value": 10,
        "step": 1,
    }

    # Create the custom element instance
    number_element = cl.CustomElement(
        name="NumberInput", 
        props=number_props
    )
    
    portals_props = {
        "title": "Portals",
        "options": ["EU", "USA", "CAN", "UK"],
        "value": "CAN"
    }

    # Create the custom element instance.
    dropdown_portals = cl.CustomElement(
        name="Dropdown",
        props=portals_props
    )
    
    url_props = {
        "title": "Model Base URL",
        "options": ["https://api.together.xyz/v1/", "https://api.openai.com/v1/"],
        "value": "https://api.together.xyz/v1/"
    }

    # Create the custom element instance.
    dropdown_url = cl.CustomElement(
        name="Dropdown",
        props=url_props
    )
    
    password_props = {
            "label": "API KEY",
            "submitted": False
        }

    # Create the custom element instance
    password_element = cl.CustomElement(
            name="PasswordInput",
            props=password_props
        )
    
    elements = [
        dropdown_url,
        password_element,
        dropdown_portals,
        number_element,
        
    ]

    # Setting elements will open the sidebar
    await cl.ElementSidebar.set_elements(elements)
    await cl.ElementSidebar.set_title("Configurations")
    # get the selected portal from the dropdown
    selected_portal_key = portals_props["value"]
    selected_url = url_props["value"]
    top_k_results = number_props["value"]
    portal_url = dict_portal.get(selected_portal_key)
    api_base_url = portal_url[: portal_url.index("api") + 3]

    
    formatted_prompt = QUERY_ANALYZER_PROMPT.format(
        api=api_base_url, portal=portal_url, top_k_results=top_k_results
    )
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": formatted_prompt}],
    )
    
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Teacher Salary Query",
            message="What is the average salary of teachers with the most working days in Canada?",
            #icon="/public/idea.svg",
            ),
        ]

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
    
    
    selector = AssistantAgent(
        "SelectorAgent",
        model_client=model_client,
        system_message="You are an agent that approves the gathered tables, if all the tables are irrelevant, you should return error otherwise you should analyze the schema of each table. Based only on the name, description and schema determine if one or multiple tables (max 2) that are needed to answer the question. Selected only the tables that can be used to answer the question, questions require 2 tables to be selected for a join.",
        output_content_type=SelectedTables,
        # TODO: add tool for join discovery @nanni00
        reflect_on_tool_use=False
    )
    
    
    prompt = f"<tables>\n{result_data['result']}\n</tables> \n<question>{question}</question>"

    # change the path of the tables to absolute path
    selected_tables = await selector.run(task=prompt)
    assert isinstance(selected_tables.messages[-1], StructuredMessage)
    assert isinstance(selected_tables.messages[-1].content, SelectedTables), "Expected the last message to be a StructuredMessage with SelectedTables content."
    
    # if selected_tables.messages[-1].content.relevance == "irrelevant":
    #     result_data["tables"] = []
    #     result_data["error"] = "All tables are irrelevant."
    #     print("All tables are irrelevant.")
    #     return result_data
    

    result_data["tables"] = convert_json_tables_to_markdown_table(selected_tables.messages[-1].content.tables)
    print(f"Selected tables: {result_data['tables']}")
    
    print(f"Selected tables: {selected_tables.messages[-1].content.tables}")

    return result_data, selected_tables.messages[-1].content.tables

@cl.step(name="Code Execution", show_input=False)
async def code_execution(selected_tables, question):
    """
    Executes the code provided in the messages and returns the result.
    This is a placeholder for actual code execution logic.
    """
    current_step = cl.context.current_step
    #current_step.output = "Executing code...\n"
    
    # semples three rows of the selected tables looking from path with csv
    import csv
    selected_tables = [table for table in selected_tables if table.path.endswith(".csv")]
    if not selected_tables:
        current_step.output = "No valid CSV tables selected."
        return "No valid CSV tables selected."

    
    # Create a prompt for the code execution agent
    # The prompt should be formatted according to the CoderCritic.md file
    # and should include the selected tables and the question.
    tables_with_rows = []
    for table in selected_tables:
        try:
            df = pd.read_csv(table.path, nrows=10)  # Read only the first 3 rows
            sample_rows = df.to_markdown()
            tables_with_rows.append({
                "title": table.title,
                "description": table.description,
                "path": table.path,
                "schema": table.schema,
                "sample_rows": sample_rows
            })
        except Exception as e:
            logger.error(f"Error reading {table}: {e}")
            continue
    
     # format tables_with_rows as a string Table: title\n Description\n Path\n Schema\n Table Example: \n
    if not tables_with_rows:
        current_step.output = "No valid tables with sample rows found."
        return "No valid tables with sample rows found."

    # format the selected tables as a string
    selected_tables_str = "\n".join(
        [f"Table: {table['title']}\nDescription: {table['description']}\nPath: {table['path']}\nSchema: {', '.join(table['schema'])}\nTable Example: \n{table['sample_rows']}"
         for table in tables_with_rows]
    )
    
    print(f"Selected tables with sample rows: {tables_with_rows}")
    
    
    prompt = CODER_CRITIC_PROMPT.format(selected_tables=selected_tables_str, question=question)


    result = ""
    with tempfile.TemporaryDirectory() as temp_dir:
        import venv
        venv_dir = "/home/angelo.mozzillo/lakegen_demo/.venv"
        venv_builder = venv.EnvBuilder(with_pip=True)
        venv_context = venv_builder.ensure_directories(venv_dir)
        executor = LocalCommandLineCodeExecutor(work_dir=temp_dir, virtual_env_context=venv_context)
        
        await executor.start()
    
        code_execution_agent = CodeExecutorAgent(
            "CodeExecutionAgent",
            code_executor=executor,
            #model_client=model_client,
            #system_message="/nothink Only if the code execution was successful return the result of the code and write STOP\n <Result>\n STOP. Otherwise return only the error message, nothing else.",
        )
        
        from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_agentchat.ui import Console
        
        coder = AssistantAgent(
            "CriticCoder",
            model_client=model_client,
            system_message="You are an expert Coder Critic specializing in evaluating dataset relevance and generating executable pandas code for data analysis tasks. Your primary responsibility is to assess whether provided tables can answer specific questions and generate robust Python solutions. Output 'STOP' after the Python script runs successful execution"
        )
        
        #termination_condition = MaxMessageTermination(6)
        text_termination = TextMentionTermination("STOP")
        #combined_termination = termination_condition | text_termination
        

        groupchat = RoundRobinGroupChat(
            participants=[coder, code_execution_agent], termination_condition=text_termination
        )

        result = groupchat.run_stream(task=prompt)
        final = ""
        async for chunk in result:
        
            if hasattr(chunk, 'source') and chunk.source == 'user':
                continue
            final += chunk.content
            await current_step.stream_token(chunk.content)

        await executor.stop()
        #await coder.stop()
        #await code_execution_agent.stop()
        
                
        # for msg in result.messages:
        #     if msg.source == 'user':
        #         continue  # Skip user messages in the output
        #     current_step.output += msg.content + "\n"
        
        
        
        answerer = AssistantAgent(
            "Answerer",
            model_client=model_client,
            system_message=f"/nothink \nYou are an assistant that provide the final answer to the user question: <question>{question}</question> on the code execution result. If the code execution was successful, return the result of the code execution in the answer to the question with the explaination on how the result is reached. If the code execution failed, return only the code in a python code block and the error message. ",
        )
        
        #print(result.messages[-1].content)
        
        # Run the answerer agent to get the final answer, append the name of the agent to the messages
        messages = ''.join([msg.source + ": " + msg.content + "\n" for msg in result.messages[:-2]])
        print(messages)
        final_answer = await Console(answerer.run_stream(task=messages))
        # If the final answer is empty, return a message indicating no code was generated
        if not final_answer.messages[-1].content.strip():
            current_step.output += "No code generated."
            return "No code generated."
        
        # take the <think> block from the final answer
        think_block = final_answer.messages[-1].content

        
        # remove the think block from the final answer
        answer = final_answer.messages[-1].content
        print(f"Final answer: {answer}")
    
    current_step.output += think_block
    return answer
    
    


@cl.step(name="Extended Thinking", show_input=False)
async def extended_thinking(messages, settings):
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
    
    print(f"Messages: {[UserMessage(source=msg['role'], content=msg['content']) for msg in messages]}")
    
    stream = model_client.create_stream(messages=[UserMessage(source=msg["role"], content=msg["content"]) for msg in messages])

    buffer = ""
    final_content = ""
    thinking_content = ""
    in_thinking_block = False
    has_thinking = False

    async for chunk in stream:
        # Ensure the incoming chunk is a string before processing.
        # This handles cases where the stream might send other data types.
        if isinstance(chunk, str):
            buffer += chunk
        else:
            # Skip chunks that are not in a recognized format.
            continue

        # Continuously process the buffer as long as there's content.
        while True:
            if not in_thinking_block:
                think_start_index = buffer.find("<think>")
                
                if think_start_index != -1:
                    # A <think> tag is found. Process the content before it as final output.
                    content_before_tag = buffer[:think_start_index]
                    if content_before_tag:
                        final_content += content_before_tag
                        # Stream this part of the final response to the user immediately.
                        await current_step.stream_token(content_before_tag)

                    # Move the buffer past the processed content and the tag.
                    buffer = buffer[think_start_index + len("<think>"):]
                    in_thinking_block = True
                    has_thinking = True
                else:
                    # **FIX**: No <think> tag found in the current buffer.
                    # Instead of waiting, stream the entire buffer as final content now.
                    if buffer:
                        final_content += buffer
                        # Stream the current buffer content to the user.
                        await current_step.stream_token(buffer)
                        buffer = "" # Clear the buffer as it has been processed.
                    break # Exit the while loop and wait for the next chunk.

            if in_thinking_block:
                think_end_index = buffer.find("</think>")

                if think_end_index != -1:
                    # A </think> tag is found. Process the content inside the tags as "thinking".
                    content_in_tag = buffer[:think_end_index]
                    if content_in_tag:
                        thinking_content += content_in_tag
                        # Stream the thinking part to the "Extended Thinking" step.
                        await current_step.stream_token(content_in_tag)

                    # Move the buffer past the processed content and the tag.
                    buffer = buffer[think_end_index + len("</think>"):]
                    in_thinking_block = False
                else:
                    # **FIX**: No </think> tag found in the current buffer.
                    # Instead of waiting, stream the entire buffer as thinking content now.
                    if buffer:
                        thinking_content += buffer
                        await current_step.stream_token(buffer)
                        buffer = "" # Clear the buffer as it has been processed.
                    break # Exit the while loop and wait for the next chunk.
    
    # This final check is now mostly for safety, as the loop handles most cases.
    if buffer:
        if in_thinking_block:
            await current_step.stream_token(buffer)
        else:
            # Stream any final leftover content.
            await current_step.stream_token(buffer)
            final_content += buffer
            
    
    return has_thinking, final_content

@cl.on_message
async def on_message(msg: cl.Message):
    """Handles incoming user messages, runs the model, and calls tools."""
    message_history = cl.user_session.get("message_history")
    question = msg.content.strip()
    message_history.append({"role": "user", "content": question})

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
        tool_res, selected_tables = await ckan_tool(url, question)

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
    # Get the model's response, which may include a tool call
   

    
    
    answer = await code_execution(selected_tables, question)

    # Stream the model's raw response to the use
    
    # Send the generated code to the user
    await cl.Message(content=answer).send()
