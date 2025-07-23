import asyncio
import os
import re

import pandas as pd
import requests
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from tqdm import tqdm

# same seed for any random operation
seed = 42

# Load environment variables from .env file
load_dotenv()
assert "TOGETHER_API_KEY" in os.environ

settings = {
    "model": "Qwen/Qwen3-235B-A22B-fp8-tput",
    "temperature": 0.1,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "max_tokens": 10000,
}


model_client = OpenAIChatCompletionClient(
    base_url="https://api.together.xyz/v1/",
    api_key=os.getenv("TOGETHER_API_KEY", ""),
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    },
    **settings,
)


portal_url = "https://open.canada.ca/data/api/3/action/package_search?q="
api_base_url = "https://open.canada.ca"


def sample_rows(df, n_keywords):
    df = pd.read_csv("keywords_gt.csv")
    df_presence = df[(df["presence"] == "1") & (df["n_keywords"] == n_keywords)]

    # Define grouping columns and perform sampling ---
    grouping_cols = ["n_keywords", "type", "difficulty"]

    # Group by the columns and sample 1 row from each group.
    sampled_df = df_presence.groupby(grouping_cols, as_index=False).sample(
        n=1, random_state=seed
    )

    # Remove sampled rows from the original dataframe ---
    remaining_df = df.drop(sampled_df.index)

    # Format the sampled data as a Markdown table ---
    # We select a subset of columns for better readability in the output.
    columns_to_display = ["nl", "keywords"]
    markdown_table = sampled_df[columns_to_display].to_markdown(index=False)

    return markdown_table, remaining_df


def check_id_presence(r_rsc_id, s_rsc_id, res_ids_set: list):
    r_id_found = s_id_found = False
    r_id_index = s_id_index = None

    if pd.notna(r_rsc_id) and r_rsc_id in res_ids_set:
        r_id_found = True
        r_id_index = res_ids_set.index(r_rsc_id)

    # Check s_rsc_id only if it's not None/NaN
    if pd.notna(s_rsc_id) and s_rsc_id in res_ids_set:
        s_id_found = True
        s_id_index = res_ids_set.index(s_rsc_id)

    return 1 if (r_id_found and (s_id_found or s_rsc_id)) else 0, r_id_index, s_id_index


async def generate_keywords(
    client: OpenAIChatCompletionClient,
    query_prompt: str,
    question: str,
    n_keywords: int,
    api_base_url: str,
    portal_url: str,
    sampled_df: str,
):
    formatted_prompt = query_prompt.format(
        api=api_base_url,
        portal=portal_url,
        n_keywords=n_keywords,  # , sampled_df=sampled_df
    )

    analyzer = AssistantAgent(
        name="QueryAnalyzer", model_client=model_client, system_message=formatted_prompt
    )

    response = await analyzer.run(task=question)
    response = response.messages[-1]

    models_usage = response.models_usage

    match = re.search(r"<keywords>(.*?)</keywords>", response.content, re.DOTALL)
    keywords = ""
    if match:
        keywords = match.group(1).strip()
        keywords = "+".join(keywords.split("+")[:n_keywords])

    return (
        keywords,
        response.content,
        models_usage.prompt_tokens,
        models_usage.completion_tokens,
    )


def fetch_ckan_results(ckan_query_url: str, keys: str, limit_value: int):
    """Fetches dataset results from a CKAN API endpoint."""
    ckan_query_url = (
        f"{ckan_query_url}q={keys}&defType=edismax&sort=sort desc&rows={limit_value}"
    )

    # Replaced the async block with a single synchronous call
    response = requests.get(ckan_query_url)
    response.raise_for_status()  # Checks for HTTP errors (e.g., 404, 500)

    # Removed 'await' from the .json() call
    data = response.json()
    results = data["result"]["results"]
    result_count = data["result"]["count"]

    # return a list of package IDs in the same
    # order as given by the API
    results_ids = [r["id"] for r in results]
    return results_ids, result_count


async def main():
    # path to the keyword extractor prompt markdown file
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "backend", "prompts", "QueryAnalyzer.md"
    )

    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    orqa_path = os.path.join(data_path, "orqa.csv")
    keywords_gt_path = os.path.join(data_path, "keywords_gt_with_examples.csv")

    assert os.path.exists(prompt_path), f"Prompt file not found there: {prompt_path}"
    assert os.path.exists(orqa_path), f"OrQA dataset not found there: {orqa_path}"

    # read the prompt template
    with open(prompt_path, "r", encoding="utf-8") as f:
        QUERY_ANALYZER_PROMPT = f.read()

    # do not use full thinking process for keyword extraction
    thinking = False
    if not thinking:
        QUERY_ANALYZER_PROMPT = "/nothink" + QUERY_ANALYZER_PROMPT

    # load the OrQA dataset
    gt = pd.read_csv(orqa_path)

    # take only canadian questions
    can = gt[gt["country_tag"] == "CAN"].reset_index(drop=True).copy()

    # do not use all the 500 questions, just a small subset for now
    limit_experiments_up_to_row = 20
    can = can.iloc[:limit_experiments_up_to_row]

    # select a subset of columns for next steps
    can_questions = can[
        [
            "country_tag",
            "type",
            "difficulty",
            "r_rsc_id",
            "s_rsc_id",
            "r_pkg_id",
            "s_pkg_id",
            "r_col_name",
            "s_col_name",
            "nl",
        ]
    ]
    total_questions = len(can_questions)

    # experiments configuration
    # one run for each number of generated keys
    list_n_keywords = [2, 3, 4, 5]
    top_k = 1000

    columns = [
        "country_tag",
        "type",
        "difficulty",
        "r_rsc_id",
        "s_rsc_id",
        "r_pkg_id",
        "s_pkg_id",
        "r_col_name",
        "s_col_name",
        "nl",
        "top_k",
        "n_keywords",
        "keywords",
        "presence",
        "result_count",
        "r_id_index",
        "s_id_index",
        "prompt_tokens",
        "completion_tokens",
        "response",
    ]

    try:
        # Read the CSV file ONCE at the start to check progress for all keys
        existing_results_df = pd.read_csv(keywords_gt_path)
        print(
            f"Found {len(existing_results_df)} existing results in '{keywords_gt_path}'."
        )
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # If file doesn't exist or is empty, create an empty DataFrame
        # and create the file with a header for subsequent appends.
        print(f"'{keywords_gt_path}' not found or is empty. Starting fresh.")
        existing_results_df = pd.DataFrame(columns=columns)
        existing_results_df.to_csv(keywords_gt_path, index=False)

    for n_keywords in list_n_keywords:
        print(f"\n{'=' * 20} Checking progress for n_keys = {n_keywords} {'=' * 20}")

        # determine Starting Point for the CURRENT n_keys_value ---
        # filter the in-memory DataFrame to count rows for this specific key
        processed_count = existing_results_df[
            existing_results_df["n_keywords"] == n_keywords
        ].shape[0]

        if processed_count >= total_questions:
            print(
                f"✅ n_keys = {n_keywords} is already complete ({processed_count}/{total_questions} rows). Skipping."
            )
            continue  # Move to the next n_keys_value

        print(
            f"Resuming n_keys = {n_keywords}. Processed: {processed_count}. Remaining: {total_questions - processed_count}."
        )

        # don't understand why a sampling of the questions is needed here
        # simply using all the questions isn't ok?
        # sampled_df, remaining_df = sample_rows(can_questions.iloc[processed_count:], n_keywords)

        # Slice the input dataframe to get only the unprocessed rows
        # questions_to_process = remaining_df
        questions_to_process = can_questions

        for index, row in tqdm(
            questions_to_process.iterrows(), total=questions_to_process.shape[0]
        ):
            try:
                # ask the model to generate n_keywords keywords for the given question
                (
                    keywords,
                    answer,
                    prompt_tokens,
                    completion_tokens,
                ) = await generate_keywords(
                    model_client,
                    QUERY_ANALYZER_PROMPT,
                    row["nl"],
                    n_keywords,
                    api_base_url,
                    portal_url,
                    None,
                )  # sampled_df)

                assert keywords, "Keys generation returned None."

                # query the CKAN API with the given keywords
                results, result_count = fetch_ckan_results(
                    "https://open.canada.ca/data/api/3/action/package_search?",
                    keywords,
                    top_k,
                )

                # check if the packages are present into the
                # fetched results, and keep track of their
                # position (the high top_k value should assure us that
                # they are found somewhere)
                presence_value, r_id_index, s_id_index = check_id_presence(
                    row["r_pkg_id"], row["s_pkg_id"], results
                )

                status_success = True

            except Exception as e:
                # --- If any operation fails, prepare an error row ---
                print(f"⚠️  Error on row index {index}: {e}. Logging failure.")
                status_success = False
                raise e

            processed_row_data = {
                "country_tag": row["country_tag"],
                "type": row["type"],
                "difficulty": row["difficulty"],
                "r_rsc_id": row["r_rsc_id"],
                "s_rsc_id": row["s_rsc_id"],
                "r_pkg_id": row["r_pkg_id"],
                "s_pkg_id": row["s_pkg_id"],
                "r_col_name": row["r_col_name"],
                "s_col_name": row["s_col_name"],
                "nl": row["nl"],
                "top_k": top_k,
                "n_keywords": n_keywords,
                "keywords": "ERROR" if not status_success else keywords,
                "presence": "ERROR" if not status_success else presence_value,
                "result_count": 0 if not status_success else result_count,
                "r_id_index": None if not status_success else r_id_index,
                "s_id_index": None if not status_success else s_id_index,
                "prompt_tokens": 0 if not status_success else prompt_tokens,
                "completion_tokens": 0 if not status_success else completion_tokens,
                "response": "ERROR" if not status_success else answer,
            }

            # --- Convert the single row to a DataFrame and append to the CSV ---
            new_row_df = pd.DataFrame([processed_row_data])
            new_row_df.to_csv(keywords_gt_path, mode="a", header=False, index=False)


if __name__ == "__main__":
    # Run the main function using asyncio
    asyncio.run(main())

