import pandas as pd
from autogen_core.models import UserMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import re
import requests 
import asyncio
from tqdm import tqdm

from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
settings = {
    "model": "Qwen/Qwen3-235B-A22B-fp8-tput",
    "temperature": 0.1,
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


portal_url = "https://open.canada.ca/data/api/3/action/package_search?q="
api_base_url = portal_url[: portal_url.index("api") + 3]
top_k_results = 10


def sample_rows(df, n_keys):
    df = pd.read_csv('key_gt.csv')
    df_presence = df[(df['presence'] == '1') & (df['n_keys'] == n_keys)]

    # --- 2. Define grouping columns and perform sampling ---
    grouping_cols = ['n_keys', 'type', 'difficulty']

    # Group by the columns and sample 1 row from each group.
    # A random_state is used for reproducibility. Remove it for different results each run.
    sampled_df = df_presence.groupby(grouping_cols, as_index=False).sample(n=1, random_state=42)

    # --- 3. Remove sampled rows from the original dataframe ---
    remaining_df = df.drop(sampled_df.index)

    # --- 4. Format the sampled data as a Markdown table ---
    # We select a subset of columns for better readability in the output.
    columns_to_display = ['nl', 'keywords']
    markdown_table = sampled_df[columns_to_display].to_markdown(index=False)
    
    return markdown_table, remaining_df



def check_id_presence(r_rsc_id, s_rsc_id, res_ids_set:list):
    r_id_found = False
    s_id_found = False
    r_id_index = None
    s_id_index = None

    if pd.notna(r_rsc_id) and r_rsc_id in res_ids_set:
        r_id_found = True
        r_id_index = res_ids_set.index(r_rsc_id)
    
    # Check s_rsc_id only if it's not None/NaN
    if pd.notna(s_rsc_id) and s_rsc_id in res_ids_set:
        s_id_found = True
        s_id_index = res_ids_set.index(s_rsc_id)
    
    return 1 if (r_id_found and (s_id_found or s_rsc_id)) else 0, r_id_index, s_id_index

async def generate_keys(client: OpenAIChatCompletionClient, query_prompt:str, question:str, n_keywords:int, api_base_url:str, portal_url:str, sampled_df:str):
    formatted_prompt = query_prompt.format(
        api=api_base_url, portal=portal_url, n_keywords=n_keywords, sampled_df=sampled_df
    )

    analyzer = AssistantAgent(
        name="QueryAnalyzer",
        model_client=model_client,
        system_message=formatted_prompt
    )
    
    response = await analyzer.run(task=question)
    response = response.messages[-1]
    #print(response.content)

    models_usage = response.models_usage

    match = re.search(r"<keywords>(.*?)</keywords>", response.content, re.DOTALL)
    keys = ""
    if match:
        keys = match.group(1).strip()
        keys = "+".join(keys.split('+')[:n_keywords])
        
            
    
    return keys, response.content, models_usage.prompt_tokens, models_usage.completion_tokens

def fetch_ckan_results(ckan_query_url: str, keys:str, limit_value:int):
    """Fetches dataset results from a CKAN API endpoint."""
    # This part remains unchanged
    #match = re.search(r'limit=(\d+)', ckan_query_url)
    #limit_value = int(match.group(1)) if match else 10
    ckan_query_url = f"{ckan_query_url}q={keys}&defType=edismax&sort=sort desc&rows={limit_value}"
    n_results = 0
    
    try:
        # Replaced the async block with a single synchronous call
        response = requests.get(ckan_query_url)
        response.raise_for_status() # Checks for HTTP errors (e.g., 404, 500)
        
        # Removed 'await' from the .json() call
        data = response.json() 
        results = data["result"]["results"]
        result_count = data["result"]["count"]

        # This part remains unchanged
        #csv_resources = [
        #    (r["title"], res.get("description", "No description"), res["url"], res["id"])
        #    for r in results[:limit_value]
        #    for res in r.get("resources", [])
        #    if res.get("format", "").strip().lower() in ["csv", "zip"]
        #    and "-fra" not in res.get("url", "").lower()
        #]
        results_ids = [r['id'] for r in results]
        return results_ids, result_count
    
    except requests.exceptions.RequestException as e:
        # It's good practice to handle potential network or HTTP errors
        print(f"An error occurred during the request: {e}")
        return []
    except KeyError:
        # Handle cases where the JSON structure is not as expected
        print("Error: Unexpected JSON structure received from the API.")
        return []


async def main():
    gt = pd.read_csv('./orqa.csv')

    filtered_gt = gt[['country_tag', 'type', 'difficulty', 'success', 'r_rsc_id', 's_rsc_id', 'r_pkg_id', 's_pkg_id', 'r_col_name',
        's_col_name', 'sql', 'nl', 'sql_success', 'tot_time']].copy()

    can = filtered_gt[filtered_gt['country_tag']=='CAN'].reset_index(drop=True).copy()


    can_questions = can[['country_tag', 'type', 'difficulty', 'r_rsc_id', 's_rsc_id', 'r_pkg_id', 's_pkg_id', 'nl']]
    total_questions = len(can_questions)
        
    thinking = False
    prompt_path = os.path.join("..", "backend", "prompts", "QueryAnalyzer.md")

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            QUERY_ANALYZER_PROMPT = f.read()
    except FileNotFoundError:
        QUERY_ANALYZER_PROMPT = "" # Assign an empty string if the file is not found

    if not thinking:
        QUERY_ANALYZER_PROMPT = '/nothink' + QUERY_ANALYZER_PROMPT

    list_n_keys = [2, 3, 4, 5]
    top_k = 1000
    key_gt_path = 'key_gt_with_examples.csv'

    columns = [
        'country_tag', 'type', 'difficulty', 'r_rsc_id', 's_rsc_id', 'r_pkg_id', 's_pkg_id','nl',
        'keywords', 'prompt_tokens', 'completion_tokens', 'presence', 'response', 'result_count', 'top_k', 'n_keys', 'r_id_index', 's_id_index'
    ]

    try:
        # Read the CSV file ONCE at the start to check progress for all keys
        existing_results_df = pd.read_csv(key_gt_path)
        print(f"Found {len(existing_results_df)} existing results in '{key_gt_path}'.")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # If file doesn't exist or is empty, create an empty DataFrame
        # and create the file with a header for subsequent appends.
        print(f"'{key_gt_path}' not found or is empty. Starting fresh.")
        existing_results_df = pd.DataFrame(columns=columns)
        existing_results_df.to_csv(key_gt_path, index=False)


    ## 3. Loop, Process, and Append Each Row
    for n_keys in list_n_keys:
        print(f"\n{'='*20} Checking progress for n_keys = {n_keys} {'='*20}")

        # --- 4. Determine Starting Point for the CURRENT n_keys_value ---
        # Filter the in-memory DataFrame to count rows for this specific key
        processed_count = existing_results_df[existing_results_df['n_keys'] == n_keys].shape[0]

        if processed_count >= total_questions:
            print(f"✅ n_keys = {n_keys} is already complete ({processed_count}/{total_questions} rows). Skipping.")
            continue  # Move to the next n_keys_value

        print(f"Resuming n_keys = {n_keys}. Processed: {processed_count}. Remaining: {total_questions - processed_count}.")

        sampled_df, remaining_df = sample_rows(can_questions.iloc[processed_count:], n_keys)
        
        # Slice the input dataframe to get only the unprocessed rows
        questions_to_process = remaining_df
        for index, row in tqdm(questions_to_process.iterrows(), total=questions_to_process.shape[0]):
            try:
                
                # --- Attempt the main operations ---
                keys, answer, prompt_tokens, completion_tokens = await generate_keys(
                    model_client, QUERY_ANALYZER_PROMPT, row['nl'], n_keys, api_base_url, portal_url, sampled_df
                )
                
                
                if not keys:
                    raise ValueError("Keys generation returned None.")
                    

                results, result_count = fetch_ckan_results("https://open.canada.ca/data/api/3/action/package_search?", keys, top_k)
                #res_ids_set = [r[3] for r in results]
                
                presence_value, r_id_index, s_id_index = check_id_presence(row['r_pkg_id'], row['s_pkg_id'], results)
                
                # Prepare the successful result as a dictionary
                processed_row_data = {
                    'country_tag': row['country_tag'], 'type': row['type'],
                    'difficulty': row['difficulty'], 'r_rsc_id': row['r_rsc_id'],
                    's_rsc_id': row['s_rsc_id'], 'r_pkg_id': row['r_pkg_id'],
                    's_pkg_id': row['s_pkg_id'], 'nl': row['nl'],
                    'keywords': keys, 'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens, 'presence': presence_value, 'response': answer, 'result_count': result_count, 'top_k': top_k, 'n_keys': n_keys, 'r_id_index': r_id_index, 's_id_index': s_id_index
                }

            except Exception as e:
                # --- If any operation fails, prepare an error row ---
                print(f"⚠️  Error on row index {index}: {e}. Logging failure.")
                processed_row_data = {
                    'country_tag': row['country_tag'], 'type': row['type'],
                    'difficulty': row['difficulty'], 'r_rsc_id': row['r_rsc_id'],
                    's_rsc_id': row['s_rsc_id'], 'r_pkg_id': row['r_pkg_id'],
                    's_pkg_id': row['s_pkg_id'], 'nl': row['nl'], # Corrected s_r_id to s_rsc_id
                    'keywords': 'ERROR', 'prompt_tokens': 0,
                    'completion_tokens': 0, 'presence': 'ERROR', 'response': 'ERROR', 'result_count': 0, 'top_k': top_k, 'n_keys': n_keys, 'r_id_index': None, 's_id_index': None
                }
            
            # --- Convert the single row to a DataFrame and append to the CSV ---
            new_row_df = pd.DataFrame([processed_row_data])
            new_row_df.to_csv(key_gt_path, mode='a', header=False, index=False)
        
if __name__ == "__main__":
    # Run the main function using asyncio
    asyncio.run(main())