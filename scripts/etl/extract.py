import requests
import logging
import os

YEAR = 2025

def extract_data() -> None:
    url = (
        "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/"
        "SRAG/2025/INFLUD25-10-11-2025.csv"
    )


    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    datalake_dir = os.path.join(project_root, "datalake", "bronze")

    os.makedirs(datalake_dir, exist_ok=True)

    output_path = os.path.join(datalake_dir, f"srag_{YEAR}.csv")
    
    response = requests.get(url)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    logging.info(f"Data saved to {output_path}")

if __name__ == "__main__":
    extract_data()