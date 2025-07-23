from io import BytesIO
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Literal
import zipfile

import pandas as pd
import requests
from .ckan import CKAN

MAX_CONTENT_LENGTH = 2 ** 30

def download_resource(id, url, download_path, download_format, verbose: bool =True):

    def csv():
        # sometimes the data are encoded, sometimes not
        # and we do not want to start reading zip files here
        assert not url.endswith('.zip')
        assert 'DOCTYPE' not in response.content[:100].decode('latin-1')
        path = os.path.join(download_path, f'{id}.{download_format}')
        
        try:
            df = pd.read_csv(response.content, **pd_read_csv_kwargs)
        except:
            # in some cases data are encoded, thus we try again with a byte io stream
            df = pd.read_csv(BytesIO(response.content), **pd_read_csv_kwargs)
        
        match download_format:
            case 'csv':
                df.to_csv(path, **pd_to_csv_kwargs)
            case 'parquet':
                df.to_parquet(path, **pd_to_parquet_kwargs)
            
    def zip():
        # open the donwloaded ZIP
        # accepted only directories with just 1 file (except any metadata file)
        with zipfile.ZipFile(BytesIO(response.content), 'r') as z:
            file_names = list(
                filter(
                    lambda fname: 'metadata' not in fname.lower() and 'fr' not in fname.lower() and fname.endswith('.csv'), 
                    z.namelist()
                )
            )

            assert len(file_names) == 1
            
            path = os.path.join(download_path, f'{id}.{download_format}')
            df = pd.read_csv(z.open(file_names[0]), **pd_read_csv_kwargs)

            match download_format:
                case 'csv':
                    df.to_csv(path, **pd_to_csv_kwargs)
                case 'parquet':
                    df.to_parquet(path, **pd_to_parquet_kwargs)

    try:
        # try to get the size of the file
        response = requests.head(url, allow_redirects=True)
        content_bytes = response.headers.get("Content-Length")

        # Accept files with limited size
        if content_bytes and int(content_bytes) > MAX_CONTENT_LENGTH:
            if verbose: print(f'Large content-length for {id=}')
            return False

        pd_read_csv_kwargs = {'sep': None, 'encoding': 'latin-1', 'encoding_errors': 'ignore', 'on_bad_lines': 'skip', 'engine': 'python'}
        pd_to_csv_kwargs = {'index': False}
        pd_to_parquet_kwargs = {'index': False, 'compression': 'gzip'}

        # download all the resource data at once
        response = requests.get(url, allow_redirects=True)
        # if verbose: print(f"Resource {id} size: {len(response.data)}")
    except TimeoutError: #urllib3.exceptions.TimeoutError:
        if verbose: print(f"Timeout with resource: {id}")
        return False

    # if logger: logger.debug(f"Extracting data from resource {rsc_name}...")

    # try each method to get the data
    success = False
    for method in [csv, zip]:
        try:             
            method()
            success = True
            break
        except Exception as e:
            # if verbose: print(f"Method {method} failed with resource {id}: {e}")
            continue
    
    # if logger: logger.debug(f"{'SUCCESS' if success else 'FAILURE'} download resource {rsc_url}")
    return success


class CanadaCKAN(CKAN):
    def __init__(self) -> None:
        super().__init__("CANADA", "https://open.canada.ca", "/data/api/3/action")

    def package_search(self, q: str | None = None, rows: int | None = None, **kwargs) -> Dict:
        return super().package_search(q=q, rows=rows, **kwargs)

    def download_tables_from_package_search(self, download_path: str, download_format: Literal['csv', 'parquet'] = 'csv', max_workers: int | None = None, **package_search_kwargs) -> int:
        results = self.package_search(**package_search_kwargs)
        results = results['result']['results']

        work = [
            (resource_metadata['id'], resource_metadata['url'], download_path, download_format)
            for package_metadata in results
            for resource_metadata in package_metadata['resources']
            if resource_metadata['format'] == 'CSV' 
            and 'en' in resource_metadata['language']
        ]

        with ProcessPoolExecutor(max_workers) as executor:
            futures = {
                executor.submit(download_resource, *task)
                for task in work
            }

            success_count = 0

            for future in futures:
                try:
                    success_count += future.result()
                except Exception as e:
                    print(e)
        
        return success_count



        


