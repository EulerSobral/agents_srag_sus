import pandas as pd
import logging
import os

def load_data(file_path: str):
    df = pd.read_csv(file_path, sep=";", on_bad_lines="skip")
    return df

def treated_data(df):
    """It processes raw data by removing irrelevant columns,
handling missing values, and eliminating duplicates."""

    columns_important = [
        'DT_NOTIFIC', 'AVE_SUINO', 'FEBRE',
        'TOSSE', 'GARGANTA', 'DISPNEIA', 'DESC_RESP',
        'DIARREIA', 'VOMITO', 'FATOR_RISC',
        'VACINA', 'ANTIVIRAL', 'TP_ANTIVIR', 'UTI',
        'TP_AMOSTRA', 'EVOLUCAO', 'VACINA_COV',
        'SURTO_SG', 'CO_DETEC'
    ]

    df_raw_important = df[columns_important]
    df_treated_nan = df_raw_important.fillna("0")
    df_treated = df_treated_nan.drop_duplicates()

    return df_treated

def save_treated_data(df_treated, output_path: str):
    df_treated.to_csv(output_path, sep=";", index=False)

if __name__ == "__main__":

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd() 

    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    bronze_path = os.path.join(project_root, "datalake", "bronze", "srag_2025.csv")
    silver_dir = os.path.join(project_root, "datalake", "silver")
    os.makedirs(silver_dir, exist_ok=True)

    silver_path = os.path.join(silver_dir, "srag_treated_2025.csv")

    df = load_data(bronze_path)
    logging.info("Data raw loaded successfully.")

    logging.info("Starting data treatment...")
    df_treated = treated_data(df)

    logging.info("Saving treated data...")
    save_treated_data(df_treated, silver_path)

    logging.info(f"File saved to: {silver_path}")