import asyncio
import json
import os
import re
import time
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from pymilvus import AnnSearchRequest, DataType, MilvusClient, Function, FunctionType, RRFRanker
from dotenv import load_dotenv
from autogen_core.models import UserMessage, SystemMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

from ckan import CanadaCKAN

load_dotenv()
assert "TOGETHER_API_KEY" in os.environ


class Evaluation(BaseModel):
    class Package(BaseModel):
        title: str
        selected: bool

    results: List[Package]

async def llm():
    keywords_gt = pd.read_csv(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'keywords_gt_with_examples_all.csv')
    )

    # take only those cases where the r package 
    # is within the first 30 results
    data = keywords_gt[
        (keywords_gt['n_keywords'] == 2) & 
        (keywords_gt['presence'] == 1) &
        (keywords_gt['r_id_index'] < 50) &
        ((keywords_gt['s_id_index'] < 50) | (pd.isna(keywords_gt['s_id_index'])))
    ].sample(100, random_state=42)

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

    with open(os.path.join('prompts', 'TablePreFilter-allatonce.md')) as file:
        TABLE_PRE_FILTER_ALL_SYSTEM_PROMPT = file.read()

    TABLE_PRE_FILTER_ALL_TASK_QUESTION_PROMPT = "<question>{question}</question>"
    TABLE_PRE_FILTER_ALL_TASK_PACKAGE_PROMPT = "<package>{title}</package><notes>{notes}</notes>"

    # analyze packages all-at-once
    analyzer_all = AssistantAgent(
        name="TablePreFilter_all", 
        model_client=model_client, 
        system_message=TABLE_PRE_FILTER_ALL_SYSTEM_PROMPT
    )

    # ask the agent for an answer. If no_think=True, the thinking process is truncated.
    # A much shorter time is required for this option
    no_think = True

    # set up our custom CKAN client
    ckan_client = CanadaCKAN() 

    # how many results we fetch from CKAN
    top_k = 30

    # how many packages the model can select
    max_selection = 2

    total_results = []

    for idx, row in tqdm(data.iterrows(), desc="All-at-once prompt analysis: ", total=data.shape[0]):
        # print(row)
        keywords = row['keywords']
        question = row['nl']

        ckan_response = ckan_client.package_search(q=keywords, rows=top_k, defType='edismax', sort='sort desc')
        packages = ckan_response['result']['results']
        package_ids = [p['id'] for p in packages]
        r_pkg_id = row['r_pkg_id']
        s_pkg_id = row['s_pkg_id']

        r_present = s_present = r_selected = s_selected = r_success = s_success = success = None
        is_valid = None
        total_time = None
        n_selected = selected_ids = None

        prompt = TABLE_PRE_FILTER_ALL_TASK_QUESTION_PROMPT
        prompt = prompt.format(question=question)

        for package in packages:
            prompt += TABLE_PRE_FILTER_ALL_TASK_PACKAGE_PROMPT.format(
                title=package['title'],
                notes=package['notes'][:200]
            ) + '\n'

        if no_think:
            prompt = '/nothink' + ' ' + prompt
        
        try:
            start_t = time.time()
            # response = await analyzer_all.run(task=prompt)
            response = await model_client.create(
                messages=[
                    SystemMessage(content=TABLE_PRE_FILTER_ALL_SYSTEM_PROMPT),
                    UserMessage(content=prompt, source='user')
                ],
                extra_create_args={"response_format": Evaluation}
            )
            end_t = time.time()
            total_time = round(end_t - start_t)

            with open(os.path.join(os.path.dirname(__file__), 'data', 'answers', f'{idx}.json'), 'w') as file:
                file.write(response.content)
            
            # # extract the answers as a list of <title, YES/NO> tuples
            # response = response.messages[-1]
            # answers = re.findall(r"<package>(.*)</package><answer>(YES|NO)</answer>", response.content)
            # answers = [(i['id'], a[0]) for i, a in zip(packages, answers) if a[1] == 'YES']
            # selected_ids = [a[0] for a in answers]

            answers = Evaluation.model_validate(json.loads(response.content))
            model_reported_results = len(answers.results)
            is_valid = model_reported_results == len(packages)
            assert is_valid
            
            selected_ids = [p['id'] for r, p in zip(answers.results, packages) if r.selected == True]

            r_present = r_pkg_id in package_ids
            s_present = s_pkg_id in package_ids and not pd.isna(s_pkg_id)

            r_selected = r_pkg_id in selected_ids
            s_selected = s_pkg_id in selected_ids

            r_success = (r_present and r_selected) or (not r_present and not r_selected and r_present is not None) 
            s_success = (s_present and s_selected) or (not s_present and not s_selected and s_present is not None)

            # success when the model recognize the correct packages
            # whenever they are present in the top-K, 
            # or when selects just one of them whether the other is missing, 
            # or when selects none if no correct package is present
            n_selected = len(selected_ids)
            success = r_success and s_success and n_selected == r_present + s_present
        except Exception:
            pass

        total_results.append(
            {
                'index': idx,
                'time(s)': total_time,
                'prompt_tokens': response.usage.prompt_tokens, # response.models_usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens, # response.models_usage.completion_tokens,
                'r_pkg_id': r_pkg_id,
                's_pkg_id': s_pkg_id,
                'top_k': top_k,
                'result_count': len(packages),
                'n_model_results': model_reported_results,
                'is_valid': is_valid,
                'r_present': r_present,
                'r_selected': r_selected,
                'r_success': r_success,
                's_present': s_present,
                's_selected': s_selected,
                's_success': s_success,
                'success': success,
                'n_selected': n_selected,
            }
        )

        if len(total_results) > 0 and len(total_results) % 10 == 0:
            pd.DataFrame(total_results) \
                .to_csv(os.path.join(os.path.dirname(__file__), 'data', 'allatonce_evaluation_k{top_k}.csv'), index=False)

    pd.DataFrame(total_results) \
        .to_csv(os.path.join(os.path.dirname(__file__), 'data', f'allatonce_evaluation_k{top_k}.csv'), index=False)


async def select_packages_subset(
        question: str, 
        packages: List[Dict], 
        model_client: OpenAIChatCompletionClient, 
        prompts_directory: str | None = None, 
        no_think: bool = False, 
        max_selection: int = 2, 
        use_notes: bool = False,
        use_keywords: bool = False,
        strict: bool = False) -> List[Dict]:
    
    if prompts_directory is None:
        prompts_directory = 'prompts'

    with open(os.path.join(prompts_directory, 'TablePreFilter-allatonce.md')) as file:
        TABLE_PRE_FILTER_ALL_SYSTEM_PROMPT = file.read()

    TABLE_PRE_FILTER_ALL_TASK_QUESTION_PROMPT = "<question>{question}</question>"
    TABLE_PRE_FILTER_ALL_TASK_PACKAGE_PROMPT = "<package>{title}</package><notes>{notes}</notes>"

    prompt = TABLE_PRE_FILTER_ALL_TASK_QUESTION_PROMPT
    prompt = prompt.format(question=question)

    for package in packages:
        prompt += TABLE_PRE_FILTER_ALL_TASK_PACKAGE_PROMPT.format(
            title=package['title'],
            notes=package['notes'][:200] if use_notes else 'N/A'
        ) + '\n'

    if no_think:
        prompt = '/nothink' + ' ' + prompt
    
    response = await model_client.create(
        messages=[
            SystemMessage(content=TABLE_PRE_FILTER_ALL_SYSTEM_PROMPT),
            UserMessage(content=prompt, source='user')
        ],
        extra_create_args={"response_format": Evaluation}
    )
            
    answers = Evaluation.model_validate(json.loads(response.content))
    model_reported_results = len(answers.results)
    is_valid = model_reported_results == len(packages)
    if strict and not is_valid:
        raise RuntimeError(f"Model output contains a incorrect number of packages: {model_reported_results}")
    
    selected_ids = [p['id'] for r, p in zip(answers.results, packages) if r.selected == True]
    filtered_ids = [p['id'] for r, p in zip(answers.results, packages) if r.selected == False]

    if strict and len(selected_ids) != max_selection:
        raise RuntimeError(f"Model selected too many tables: {len(selected_ids)}")

    return selected_ids, filtered_ids




from together import Together
import pymilvus
from scipy.spatial.distance import cosine

def distance(x, y):
    return cosine(x, y)


def get_embedding(s: str, client: Together):
    response = client.embeddings.create(
        model="BAAI/bge-large-en-v1.5",
        input = s
    )

    return list(response.data[0].embedding)


from pymilvus.model.hybrid import BGEM3EmbeddingFunction

def bge_embedding(s: str, embedder: BGEM3EmbeddingFunction):
    e = embedder.encode_documents([s])
    return e['dense'][0]
    

async def embedding():
    keywords_gt = pd.read_csv(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'keywords_gt_with_examples_all.csv')
    )

    # take only those cases where the r package 
    # is within the first 30 results
    data = keywords_gt[
        (keywords_gt['n_keywords'] == 2) & 
        (keywords_gt['presence'] == 1) &
        (keywords_gt['r_id_index'] < 50) &
        ((keywords_gt['s_id_index'] < 50) | (pd.isna(keywords_gt['s_id_index'])))
    ].sample(100, random_state=42)

    # set up our custom CKAN client
    ckan_client = CanadaCKAN() 

    together_client = Together()
    
    # how many results we fetch from CKAN
    top_k = 30

    # how many packages the model can select
    max_selection = 2

    total_results = []

    for idx, row in tqdm(data.iterrows(), desc="All-at-once prompt analysis: ", total=data.shape[0]):
        # print(row)
        keywords = row['keywords']
        question = row['nl']

        e_question = get_embedding(question, together_client)

        ckan_response = ckan_client.package_search(q=keywords, rows=top_k, defType='edismax', sort='sort desc')
        packages = ckan_response['result']['results']
        package_ids = [p['id'] for p in packages]
        r_pkg_id = row['r_pkg_id']
        s_pkg_id = row['s_pkg_id']

        r_present = s_present = r_selected = s_selected = r_success = s_success = success = None
        is_valid = None
        total_time = None
        n_selected = selected_ids = None

        e_packages = {
            p['id']: get_embedding(s=f"{p['title']} {p['notes'][:300]}", client=together_client)
            for p in packages
        }

        with open('embeddings.json', 'w') as file:
            json.dump(e_packages, file)

        try:
            start_t = time.time()
            results = []
            for package_id, e in e_packages.items():
                results.append((package_id, distance(e_question, e)))

            best_distances = [x[1] for x in sorted(results, key=lambda x: x[1])[:2]]
            selected_ids = [x[0] for x in sorted(results, key=lambda x: x[1])[:2]]

            end_t = time.time()
            total_time = round(end_t - start_t)
            
            r_present = r_pkg_id in package_ids
            s_present = s_pkg_id in package_ids and not pd.isna(s_pkg_id)

            r_selected = r_pkg_id in selected_ids
            s_selected = s_pkg_id in selected_ids

            r_success = (r_present and r_selected) or (not r_present and not r_selected and r_present is not None) 
            s_success = (s_present and s_selected) or (not s_present and not s_selected and s_present is not None)

            # success when the model recognize the correct packages
            # whenever they are present in the top-K, 
            # or when selects just one of them whether the other is missing, 
            # or when selects none if no correct package is present
            n_selected = len(selected_ids)
            success = r_success and s_success and n_selected == r_present + s_present
        except Exception as e:
            raise e

        total_results.append(
            {
                'index': idx,
                'time(s)': total_time,
                # 'prompt_tokens': response.usage.prompt_tokens, # response.models_usage.prompt_tokens,
                # 'completion_tokens': response.usage.completion_tokens, # response.models_usage.completion_tokens,
                'distance': 'cosine',
                'best_dist': best_distances,
                'r_pkg_id': r_pkg_id,
                's_pkg_id': s_pkg_id,
                'top_k': top_k,
                'result_count': len(packages),
                # 'n_model_results': model_reported_results,
                'is_valid': is_valid,
                'r_present': r_present,
                'r_selected': r_selected,
                'r_success': r_success,
                's_present': s_present,
                's_selected': s_selected,
                's_success': s_success,
                'success': success,
                'n_selected': n_selected,
            }
        )

        if len(total_results) > 0 and len(total_results) % 10 == 0:
            pd.DataFrame(total_results) \
                .to_csv(os.path.join(os.path.dirname(__file__), 'data', f'embedding_eval_k{top_k}.csv'), index=False)

    pd.DataFrame(total_results) \
        .to_csv(os.path.join(os.path.dirname(__file__), 'data', f'embedding_eval_k{top_k}.csv'), index=False)



def define_schema(client: MilvusClient, embedding_dim: int):
    schema = client.create_schema(auto_id=False)

    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=1000, is_primary=True, description="package id")
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000, enable_analyzer=True, description="title+question")
    schema.add_field(field_name="text_dense", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim, description="text dense embedding")
    schema.add_field(field_name="text_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR, description="text sparse embedding auto-generated by the built-in BM25 function")
    
    # Add function to schema
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["text_sparse"],
        function_type=FunctionType.BM25,
    )

    schema.add_function(bm25_function)
    return schema


def create_index(client: MilvusClient):
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="text_dense",
        index_name="text_dense_index",
        index_type="AUTOINDEX",
        metric_type="IP"
    )

    index_params.add_index(
        field_name="text_sparse",
        index_name="text_sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE"}, # or "DAAT_WAND" or "TAAT_NAIVE"
    )

    return index_params



def main():
    milvus_client = pymilvus.MilvusClient('./milvus_embeddings.db')
    define_schema(milvus_client)



def milvus_embedding():
    keywords_gt = pd.read_csv(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'keywords_gt_with_examples_all.csv')
    )

    # take only those cases where the r package 
    # is within the first 30 results
    data = keywords_gt[
        (keywords_gt['n_keywords'] == 2) & 
        (keywords_gt['presence'] == 1) &
        (keywords_gt['r_id_index'] < 50) &
        ((keywords_gt['s_id_index'] < 50) | (pd.isna(keywords_gt['s_id_index'])))
    ].sample(100, random_state=42)

    # set up our custom CKAN client
    ckan_client = CanadaCKAN() 

    # together_client = Together()
    # 
    # response = together_client.embeddings.create(
    #     model="BAAI/bge-large-en-v1.5",
    #     input="Hi"
    # )
    # embedding_dim = len(response.data[0].embedding)
    
    bge_m3_ef = BGEM3EmbeddingFunction(
        model_name='BAAI/bge-m3', # Specify the model name
        device='cpu', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
    )

    embedding_dim = bge_m3_ef.dim

    milvus_client = pymilvus.MilvusClient('./milvus_embeddings.db')
    
    schema = define_schema(milvus_client, embedding_dim)
    index_params = create_index(milvus_client)

    # how many results we fetch from CKAN
    top_k = 30

    # how many packages the model can select
    max_selection = 2

    total_results = []

    for idx, row in tqdm(data.iterrows(), desc="All-at-once prompt analysis: ", total=data.shape[0]):
        milvus_client.drop_collection('packages_collection')
        milvus_client.create_collection(
            collection_name="packages_collection",
            schema=schema,
            index_params=index_params,
            overwrite=True
        )

        # print(row)
        keywords = row['keywords']
        question = row['nl']

        e_question = bge_embedding(question, bge_m3_ef)

        ckan_response = ckan_client.package_search(q=keywords, rows=top_k, defType='edismax', sort='sort desc')
        packages = ckan_response['result']['results']
        package_ids = [p['id'] for p in packages]
        r_pkg_id = row['r_pkg_id']
        s_pkg_id = row['s_pkg_id']

        r_present = s_present = r_selected = s_selected = r_success = s_success = success = None
        is_valid = None
        total_time = None
        n_selected = selected_ids = None

        docs = [
            {
                'id': p['id'],
                'text': f"{p['title']} {p['notes']}",
                'text_dense': bge_embedding(f"{p['title']} {p['notes']}", bge_m3_ef)
            }
            for p in packages
        ]

        results = milvus_client.insert(
            collection_name="packages_collection",
            data=docs
        )

        try:
            start_t = time.time()
            
            # text semantic search (dense)
            search_param_1 = {
                "data": [e_question],
                "anns_field": "text_dense",
                "param": {"nprobe": 10},
                "limit": 2
            }
            request_1 = AnnSearchRequest(**search_param_1)

            # full-text search (sparse)
            search_param_2 = {
                "data": [question],
                "anns_field": "text_sparse",
                "param": {"drop_ratio_search": 0.2},
                "limit": 2
            }
            request_2 = AnnSearchRequest(**search_param_2)

            reqs = [request_1, request_2]
            
            ranker = RRFRanker(100)

            results = milvus_client.hybrid_search(
                collection_name='packages_collection',
                reqs=reqs,
                ranker=ranker,
                limit=2
            )[0]

            selected_ids = [x['id'] for x in results]

            end_t = time.time()
            total_time = round(end_t - start_t)
            
            r_present = r_pkg_id in package_ids
            s_present = s_pkg_id in package_ids and not pd.isna(s_pkg_id)

            r_selected = r_pkg_id in selected_ids
            s_selected = s_pkg_id in selected_ids

            r_success = (r_present and r_selected) or (not r_present and not r_selected and r_present is not None) 
            s_success = (s_present and s_selected) or (not s_present and not s_selected and s_present is not None)

            # success when the model recognize the correct packages
            # whenever they are present in the top-K, 
            # or when selects just one of them whether the other is missing, 
            # or when selects none if no correct package is present
            n_selected = len(selected_ids)
            success = r_success and s_success and n_selected == r_present + s_present
        except Exception as e:
            raise e

        total_results.append(
            {
                'index': idx,
                'time(s)': total_time,
                # 'prompt_tokens': response.usage.prompt_tokens, # response.models_usage.prompt_tokens,
                # 'completion_tokens': response.usage.completion_tokens, # response.models_usage.completion_tokens,
                'ranker': 'rrf',
                'r_pkg_id': r_pkg_id,
                's_pkg_id': s_pkg_id,
                'top_k': top_k,
                'result_count': len(packages),
                # 'n_model_results': model_reported_results,
                'is_valid': is_valid,
                'r_present': r_present,
                'r_selected': r_selected,
                'r_success': r_success,
                's_present': s_present,
                's_selected': s_selected,
                's_success': s_success,
                'success': success,
                'n_selected': n_selected,
            }
        )

        if len(total_results) > 0 and len(total_results) % 10 == 0:
            pd.DataFrame(total_results) \
                .to_csv(os.path.join(os.path.dirname(__file__), 'data', f'milvus_eval_k{top_k}.csv'), index=False)

    pd.DataFrame(total_results) \
        .to_csv(os.path.join(os.path.dirname(__file__), 'data', f'milvus_eval_k{top_k}.csv'), index=False)



if __name__ == '__main__':
    # asyncio.run(llm())
    # asyncio.run(embedding())
    milvus_embedding()
