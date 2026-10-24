""" 
Name: data_pulling.py 
Author: Owen Sharpe 
Description: Pulls a pinned version of the raw NBA dataset from Kaggle, copies it into the `raw_data` folder, 
writes a manifest recording what was pulled, and prints each file's columns for a quick check.
""" 


# import libraries 
import kagglehub 
import pandas as pd
import pyarrow.parquet as pq


import json
import shutil
from datetime import date
from pathlib import Path


# global variables
DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"
DATA_VERSION = 520  # change to latest available at the most recent data pull if we need updated statistics
RAW_DIR = Path(__file__).parent / "raw_data"


# download dataset
def download_data(dataset, version):
    download_path = f"{DATASET}/versions/{DATA_VERSION}"
    return Path(kagglehub.dataset_download(download_path))

# copy cached files into the `raw_data` folder
def copy_to_raw(src_path, folder_path):
    shutil.copytree(src_path, folder_path, dirs_exist_ok=True)

# write json metadata
def write_manifest(folder_path, dataset, version):
    files = {
        f.name: f.stat().st_size
        for f in sorted(folder_path.rglob("*"))
        if f.is_file()
        and f.name != "manifest.json"
        and not f.name.startswith(".")
    }

    manifest = {
        "dataset": dataset,
        "version": version,
        "pulled_on": str(date.today()),
        "files_bytes": files,
    }

    with open(folder_path / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

# helper function for `inspect`
def preview(file, n_rows=5):
    if file.suffix == ".csv":
        return pd.read_csv(file, nrows=n_rows)
    elif file.suffix == ".parquet":
        batch = next(pq.ParquetFile(file).iter_batches(batch_size=n_rows))
        return batch.to_pandas()
    else:
        return None

# inspecting data files
def inspect(folder_path):
    pd.set_option("display.max_columns", None)
    for file in sorted(folder_path.rglob("*")):
        df = preview(file)
        
        if df is None:
            continue

        print(f"\n{file.name} ({file.stat().st_size / 1e6:.0f} MB)")
        print(df.columns.tolist())
        print(df.head())


def main(): 

    # download destination
    dest = RAW_DIR / f"v{DATA_VERSION}"

    # skip download entirely and just copy if the version already exists in storage
    if (dest/ "manifest.json").exists():
        print(f"Version {DATA_VERSION} already in {dest}, skipping download.")
    else:
        cache_file_path = download_data(DATASET, DATA_VERSION)
        copy_to_raw(cache_file_path, dest)
        write_manifest(dest, DATASET, DATA_VERSION)
        print(f"Copied version {DATA_VERSION} to {dest}")

        inspect(dest)

    
if __name__ == "__main__": 
    main()
