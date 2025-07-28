from io import BytesIO
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Literal
import zipfile
import re
import unicodedata

import pandas as pd
import requests
from tqdm import tqdm
from .ckan import CKAN

MAX_CONTENT_LENGTH = 2 ** 30

def csv(url, data, resource_id, download_path, download_format, 
        pd_read_csv_kwargs, pd_to_csv_kwargs, pd_to_parquet_kwargs):
    # sometimes the data are encoded, sometimes not
    # and we do not want to start reading zip files here
    assert not url.endswith('.zip')
    assert 'DOCTYPE' not in data[:100].decode('latin-1')
    path = os.path.join(download_path, f'{resource_id}.{download_format}')
    
    try:
        df = pd.read_csv(data, **pd_read_csv_kwargs)
    except:
        # in some cases data are encoded, thus we try again with a byte io stream
        df = pd.read_csv(BytesIO(data), **pd_read_csv_kwargs)
    
    match download_format:
        case 'csv':
            df.to_csv(path, **pd_to_csv_kwargs)
        case 'parquet':
            df.to_parquet(path, **pd_to_parquet_kwargs)
        
def zip(url, data, resource_id, download_path, download_format, 
        pd_read_csv_kwargs, pd_to_csv_kwargs, pd_to_parquet_kwargs):
    # open the donwloaded ZIP
    # accepted only directories with just 1 file (except any metadata file)
    with zipfile.ZipFile(BytesIO(data), 'r') as z:
        file_names = list(
            filter(
                lambda fname: 'metadata' not in fname.lower() and 'fr' not in fname.lower() and fname.endswith('.csv'), 
                z.namelist()
            )
        )

        assert len(file_names) == 1
        
        path = os.path.join(download_path, f'{resource_id}.{download_format}')
        df = pd.read_csv(z.open(file_names[0]), **pd_read_csv_kwargs)

        match download_format:
            case 'csv':
                df.to_csv(path, **pd_to_csv_kwargs)
            case 'parquet':
                df.to_parquet(path, **pd_to_parquet_kwargs)
                

def normalize_column_name(column_name):
    """
    Transform a column name into a SQL-safe identifier.
    """
    # Convert to lowercase
    name = column_name.lower()
    
    # Handle specific character replacements
    replacements = {
        ' ': '_',
        '/': '_/_',  # Keep the slash pattern
        '&': '_and_',
        '%': '_percent_',
        '#': '_hash_',
        '@': '_at_',
        '$': '_dollar_',
        '+': '_plus_',
        '-': '_',
        '(': '_',
        ')': '_',
        '[': '_',
        ']': '_',
        '{': '_',
        '}': '_',
        '|': '_',
        '\\': '_',
        ':': '_',
        ';': '_',
        '"': '_',
        "'": '_',
        '<': '_',
        '>': '_',
        ',': '_',
        '.': '_',
        '?': '_',
        '!': '_',
        '=': '_equals_',
        '*': '_',
    }
    
    # Apply replacements
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    return name

def normalize_dataframe_columns(df):
    """
    Normalize all column names in a DataFrame.
    """
    # Create a mapping of original to normalized names
    column_mapping = {col: normalize_column_name(str(col)) for col in df.columns}
    
    # Handle duplicate normalized names by adding suffixes
    normalized_names = list(column_mapping.values())
    seen = {}
    for original, normalized in column_mapping.items():
        if normalized in seen:
            # Add suffix for duplicates
            counter = seen[normalized] + 1
            seen[normalized] = counter
            column_mapping[original] = f"{normalized}_{counter}"
        else:
            seen[normalized] = 0
    
    # Rename columns
    df_normalized = df.rename(columns=column_mapping)
    return df_normalized, column_mapping

def csv_with_normalization(url, data, id, download_path, download_format, 
                          pd_read_csv_kwargs, pd_to_csv_kwargs, pd_to_parquet_kwargs):
    """
    Modified CSV processing function that normalizes column names.
    """
    from io import BytesIO
    
    # Read CSV data
    df = pd.read_csv(BytesIO(data), **pd_read_csv_kwargs)
    
    # Normalize column names
    df_normalized, column_mapping = normalize_dataframe_columns(df)
    
    # Save the normalized DataFrame
    if download_format == 'csv':
        output_path = f"{download_path}/{id}.csv"
        df_normalized.to_csv(output_path, **pd_to_csv_kwargs)
    elif download_format == 'parquet':
        output_path = f"{download_path}/{id}.parquet"
        df_normalized.to_parquet(output_path, **pd_to_parquet_kwargs)
    
    return df_normalized, column_mapping

def zip_with_normalization(url, data, id, download_path, download_format,
                          pd_read_csv_kwargs, pd_to_csv_kwargs, pd_to_parquet_kwargs):
    """
    Modified ZIP processing function that normalizes column names.
    """
    import zipfile
    from io import BytesIO
    
    with zipfile.ZipFile(BytesIO(data)) as zip_file:
        # Find CSV files in the zip
        csv_files = [name for name in zip_file.namelist() if name.endswith('.csv')]
        
        if not csv_files:
            raise ValueError("No CSV files found in ZIP archive")
        
        # Process the first CSV file found
        csv_file = csv_files[0]
        with zip_file.open(csv_file) as file:
            df = pd.read_csv(file, **pd_read_csv_kwargs)
            
            # Normalize column names
            df_normalized, column_mapping = normalize_dataframe_columns(df)
            
            # Save the normalized DataFrame
            if download_format == 'csv':
                output_path = f"{download_path}/{id}.csv"
                df_normalized.to_csv(output_path, **pd_to_csv_kwargs)
            elif download_format == 'parquet':
                output_path = f"{download_path}/{id}.parquet"
                df_normalized.to_parquet(output_path, **pd_to_parquet_kwargs)
    
    return df_normalized, column_mapping

def download_resource_normalized(id, name, url, download_path, download_format, 
                     verbose: bool = True, normalize_columns: bool = True):
    """
    Download resource with optional column name normalization.
    
    Args:
        id: Resource identifier
        name: Resource name
        url: Download URL
        download_path: Path to save the file
        download_format: Format to save ('csv' or 'parquet')
        verbose: Print verbose output
        normalize_columns: Whether to normalize column names
    
    Returns:
        dict: {'success': bool, 'column_mapping': dict, 'dataframe': pd.DataFrame}
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0'
    }
    
    try:
        # Try to get the size of the file
        response = requests.head(url, headers=headers, allow_redirects=True)
        content_bytes = response.headers.get("Content-Length")
        
        # Accept files with limited size
        if content_bytes and int(content_bytes) > MAX_CONTENT_LENGTH:
            if verbose: 
                print(f'Large content-length for {id=}')
            return {'success': False, 'column_mapping': None, 'dataframe': None}
        
        pd_read_csv_kwargs = {
            'sep': None, 
            'encoding': 'latin-1', 
            'encoding_errors': 'ignore', 
            'on_bad_lines': 'skip', 
            'engine': 'python'
        }
        pd_to_csv_kwargs = {'index': False}
        pd_to_parquet_kwargs = {'index': False, 'compression': 'gzip'}
        
        # Download all the resource data at once
        data = requests.get(url, timeout=15, headers=headers, allow_redirects=True).content
        
    except TimeoutError:
        return {'success': False, 'column_mapping': None, 'dataframe': None}
    
    # Choose methods based on normalization preference
    if normalize_columns:
        methods = [csv_with_normalization, zip_with_normalization]
    else:
        methods = [csv, zip]  # Your original methods
    
    # Try each method to get the data
    for method in methods:
        try:
            if normalize_columns:
                df_normalized, column_mapping = method(
                    url, data, id, download_path, download_format,
                    pd_read_csv_kwargs, pd_to_csv_kwargs, pd_to_parquet_kwargs
                )
                
                if verbose:
                    print(f"Successfully downloaded and normalized {id}")
                    print(f"Column mapping: {column_mapping}")
                
                return {
                    'success': True, 
                    'column_mapping': column_mapping, 
                    'dataframe': df_normalized
                }
            else:
                method(url, data, id, download_path, download_format,
                      pd_read_csv_kwargs, pd_to_csv_kwargs, pd_to_parquet_kwargs)
                return {'success': True, 'column_mapping': None, 'dataframe': None}
                
        except Exception as e:
            if verbose: 
                print(f"Method {method.__name__} failed with resource {id}: {e}")
            continue
    
    return {'success': False, 'column_mapping': None, 'dataframe': None}

    """
    Transform a column name into a SQL-safe identifier.
    
    Rules:
    - Convert to lowercase
    - Replace spaces with underscores
    - Handle special characters and accents
    - Keep forward slashes as underscores
    - Remove or replace other special characters
    """
    # Convert to lowercase
    name = column_name.lower()
    
    # Handle specific character replacements
    replacements = {
        ' ': '_',
        '/': '_/_',  # Keep the slash pattern as shown in your example
        '&': '_and_',
        '%': '_percent_',
        '#': '_hash_',
        '@': '_at_',
        '$': '_dollar_',
        '+': '_plus_',
        '-': '_',
        '(': '_',
        ')': '_',
        '[': '_',
        ']': '_',
        '{': '_',
        '}': '_',
        '|': '_',
        '\\': '_',
        ':': '_',
        ';': '_',
        '"': '_',
        "'": '_',
        '<': '_',
        '>': '_',
        ',': '_',
        '.': '_',
        '?': '_',
        '!': '_',
        '=': '_equals_',
        '*': '_',
    }
    
    # Apply replacements
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Handle accented characters - keep them as they appear in your example
    # The example shows "minista_re" which suggests keeping the special chars
    # but if you want to normalize accents, uncomment the next lines:
    # name = unicodedata.normalize('NFD', name)
    # name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    return name
    

def download_resource(id, name, url, download_path, download_format, verbose: bool =True):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0'
    }

    try:
        # try to get the size of the file
        response = requests.head(url, headers=headers, allow_redirects=True)
        content_bytes = response.headers.get("Content-Length")

        # Accept files with limited size
        if content_bytes and int(content_bytes) > MAX_CONTENT_LENGTH:
            if verbose: print(f'Large content-length for {id=}')
            return False

        pd_read_csv_kwargs = {'sep': None, 'encoding': 'latin-1', 'encoding_errors': 'ignore', 'on_bad_lines': 'skip', 'engine': 'python'}
        pd_to_csv_kwargs = {'index': False}
        pd_to_parquet_kwargs = {'index': False, 'compression': 'gzip'}

        # download all the resource data at once
        data = requests.get(url, timeout=15, headers=headers, allow_redirects=True).content
    except TimeoutError:
        return False

    # try each method to get the data
    success = False
    for method in [csv, zip]:
        try:             
            method(url, data, id, download_path, download_format, 
                   pd_read_csv_kwargs, pd_to_csv_kwargs, pd_to_parquet_kwargs)
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

    def download_tables_from_package_search(
            self, 
            download_path: str, 
            download_format: Literal['csv', 'parquet'] = 'csv', 
            max_workers: int | None = None, 
            verbose: bool = False,
            **package_search_kwargs
            ) -> int:
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

            for future in tqdm(futures, desc="Fetching resources: ", disable=not verbose):
                try:
                    success_count += future.result()
                except Exception as e:
                    print(e)

        return work, success_count
    
    def download_tables_from_selected_packages(
            self,
            selected_packages: list[Dict[str, str]],    
            download_path: str,
            download_format: Literal['csv', 'parquet'] = 'csv',
            max_workers: int | None = None,
            verbose: bool = False
            ) -> int:
        
        #print(f"Selected packages: {selected_packages}")  # Debugging line
        
        work = [
            (first_csv['id'], first_csv['name'], first_csv['url'], download_path, download_format)
            for package_metadata in selected_packages
            if (first_csv := next(
                (resource for resource in package_metadata['resources']
                if resource['format'] == 'CSV' and 'en' in resource['language']),
                None
            )) is not None
        ]
            
        with ProcessPoolExecutor(max_workers) as executor:
            futures = {
                executor.submit(download_resource_normalized, *task)
                for task in work
            }

            success_count = 0

            for future in tqdm(futures, desc="Fetching resources: ", disable=not verbose):
                try:
                    success_count += future.result()
                except Exception as e:
                    print(e)

        return work, success_count
