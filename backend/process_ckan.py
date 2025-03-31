import aiohttp
import asyncio
import os
import zipfile
import io
import re
import csv
import time
import requests
import logging
import chardet
from datetime import datetime
import glob

# ✅ Configure logging
# Define log file name with timestamp
LOG_DIR = "logs/agents"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"process_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE)]
)
logger = logging.getLogger(__name__)

# Cleanup old logs, keeping only the last 3
def cleanup_old_logs():
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "process_*.log")), reverse=True)
    for old_log in log_files[3:]:  # Keep only the latest 3 logs
        os.remove(old_log)

cleanup_old_logs()

# Example logging
logger.info("This is a new log entry.")

CKAN_QUERY_URL = "https://open.canada.ca/data/api/3/action/package_search?q=households+dwelling+canada;limit=10"
DOWNLOAD_FOLDER = "downloads"

# Function to clear the downloads folder
def clear_downloads():
    if os.path.exists(DOWNLOAD_FOLDER):
        csv_files = [f for f in os.listdir(DOWNLOAD_FOLDER) if f.endswith(".csv")]
        # Sort files by modification time and keep only the last 10
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(DOWNLOAD_FOLDER, x)), reverse=True)
        files_to_delete = csv_files[10:]

        for file in files_to_delete:
            os.remove(os.path.join(DOWNLOAD_FOLDER, file))
            logger.info(f"🗑️ Deleted old file: {file}")
    else:
        os.makedirs(DOWNLOAD_FOLDER)
    logger.info(f"✅ Download folder maintained with last 10 CSVs.")


# Async function to fetch CKAN data
async def fetch_ckan_results(ckan_query_url: str):
    #find limit in the url and take the value
    match = re.search(r'limit=(\d+)', ckan_query_url)
    limit_value = int(match.group(1)) if match else None
    # Remove everything after and including the first semicolon
    ckan_query_url = re.sub(r';.*', '', ckan_query_url)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(ckan_query_url) as response:
            response.raise_for_status()
            data = await response.json()
            results = data["result"]["results"]

            # Extract CSV and ZIP resources
            csv_resources = [
                (r["title"], res["description"], res["url"])
                for r in results[:limit_value]
                for res in r.get("resources", [])
                if (res.get("format", "").strip().lower() in ["csv", "zip"])
                and ("-fra" not in res.get("url", "").lower())
            ]
            
            logger.info(f"✅ Found {len(csv_resources)} CSV resources")
            return csv_resources[:limit_value]

# Async function to fetch schema from CSV or ZIP
async def fetch_schema(session, url):
    """Attempts to fetch schema of CSV inside a ZIP or a standalone CSV."""
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            if url.endswith(".zip"):
                logger.info(f"📦 Processing ZIP: {url}")
                zip_content = await response.read()
                with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
                    # ✅ Use a generator to filter CSVs lazily (faster than list comprehension)
                    csv_file = next((name for name in z.namelist() if name.lower().endswith(".csv") and "_MetaData" not in name), None)
                    if not csv_file:
                        return None

                    # ✅ Open and read only the first line efficiently
                    with z.open(csv_file) as f:
                        raw_data = f.read(1024)  # Read a portion of the file
                        detected = chardet.detect(raw_data)
                        encoding = detected['encoding']
                        print(f"Detected encoding: {encoding}")
                        f.seek(0)

                        # Use the detected encoding
                        first_line = next(csv.reader(io.TextIOWrapper(f, encoding=encoding)), None)

                    return url, csv_file, first_line  # Return CSV filename and first row as schema
            elif 'csv' in url:
                logger.info(f"📄 Processing CSV: {url}")
                first_line = await response.text()
                return url, None, str(first_line.splitlines()[0].split(","))  # First row as schema
            else:
                return None
    except Exception as e:
        logger.error(f"❌ Schema fetch failed for {url}: {e}")
        return None

# ✅ Function to extract CSV from ZIP in a separate thread (uses `requests`)
def extract_csv_from_zip(zip_url):
    """Downloads a ZIP file, extracts the main CSV (ignoring metadata), and saves it if not already downloaded."""
    try:
        logger.info(f"⬇️ Downloading ZIP: {zip_url}")
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Filter CSVs, ignoring metadata files
            csv_files = [name for name in z.namelist() if name.lower().endswith(".csv") and "_MetaData" not in name]
            if not csv_files:
                logger.error(f"❌ No valid CSV found in ZIP: {zip_url}")
                return

            # Extract the first valid CSV
            csv_filename = csv_files[0]
            save_path = os.path.join(DOWNLOAD_FOLDER, os.path.basename(csv_filename))

            # Check if the CSV already exists in the download folder
            if os.path.exists(save_path):
                logger.info(f"❌ CSV {csv_filename} already exists in {DOWNLOAD_FOLDER}. Skipping extraction.")
                return

            with z.open(csv_filename) as src, open(save_path, "wb") as f:
                f.write(src.read())  # Write the contents of the CSV to the file
            
            logger.info(f"✅ Extracted {csv_filename} from ZIP: {zip_url} to {save_path}")

    except Exception as e:
        logger.error(f"❌ Failed to extract CSV from ZIP: {e}")
# ✅ Function to extract CSV from ZIP in an async-friendly way
async def extract_csv_from_zip_async(zip_url):
    """Runs the extract_csv_from_zip function asynchronously without blocking the event loop."""
    return await asyncio.to_thread(extract_csv_from_zip, zip_url)

async def process_datasets(ckan_query_url: str):
    """Fetch datasets, process schemas, and download ZIP contents asynchronously."""
    try:
        clear_downloads()  # Ensure the download folder is empty

        start_results = time.time()  # Start timing results processing
        # Step 1: Fetch CKAN data
        csv_resources = await fetch_ckan_results(ckan_query_url)

        async with aiohttp.ClientSession() as session:
            # Step 2: Fetch schemas concurrently
            schema_results = await asyncio.gather(*[fetch_schema(session, url) for _, _, url in csv_resources])
            
            results = []
            zip_tasks = []  # Store async tasks for ZIP extraction

            # Step 3: Process results
            for (title, desc, url), schema_data in zip(csv_resources, schema_results):
                if schema_data:  
                    url, csv_filename, schema = schema_data  

                    if csv_filename:
                        dataset_url = os.path.abspath(os.path.join(DOWNLOAD_FOLDER, os.path.basename(csv_filename)))
                    else:
                        dataset_url = url  # Keep original for standalone CSVs

                    results.append({
                        "title": title,
                        "description": desc,
                        "path": dataset_url,
                        "schema": schema,
                    })

                    # If it's a ZIP, schedule extraction asynchronously
                    if csv_filename:
                        zip_tasks.append(extract_csv_from_zip_async(url))

        end_results = time.time()
        time_results = end_results - start_results
        logger.info(f"⏱️ Time for fetching & processing results: {time_results:.2f}s")

        # Step 4: Print results before starting downloads
        logger.info(f"✅ Processed {len(results)} datasets with schemas")

        start_downloads = time.time()

        # Step 5: Run all ZIP extractions asynchronously and wait for them to complete
        await asyncio.gather(*zip_tasks)

        end_downloads = time.time()
        time_downloads = end_downloads - start_downloads
        total_time = end_downloads - start_results

        logger.info(f"⏱️ Time for ZIP downloads & extraction: {time_downloads:.2f}s")
        logger.info(f"⏱️ Total processing time (results + downloads): {total_time:.2f}s")
        logger.info("✅ All ZIP extractions completed. Exiting...")

        return {"result": results, "error": None}

    except Exception as e:
        logger.error(f"❌ Error during dataset processing: {e}")
        return {"result": [], "error": str(e)}

# Run the async process
if __name__ == "__main__":
    asyncio.run(process_datasets(CKAN_QUERY_URL))