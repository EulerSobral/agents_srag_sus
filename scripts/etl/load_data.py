import pandas as pd
import logging
import os

def final_processing_data(file_path):

    """ Removes numeric values ​​and converts them to more descriptive categorical values.
It also converts the date column to the YYYY-MM-DD format.
    """

    logging.info(f"Init process. Reading files: {file_path}")

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    gold_dir = os.path.join(project_root, "datalake", "gold")
    os.makedirs(gold_dir, exist_ok=True)

    output_path = os.path.join(gold_dir, "srag_2025_final_processed.csv")

    try:
        df = pd.read_csv(file_path, sep=";", on_bad_lines="skip")
        logging.info("Files read successfully.")
    except Exception as e:
        logging.info(f"Error reading files: {e}")
        return

    map_yes_no = {
        0: "Campo vazio", 1: "Sim", 2: "Não", 9: "Ignorado"
    }
    map_birds_pigs = {
        0: "Campo vazio", 1: "Sim, aves e/ou suínos", 2: "Não, nenhum",
        3: "Sim, outros", 9: "Ignorado"
    }
    map_evolucao = {
        0: "Campo vazio", 1: "Cura", 2: "Óbito", 3: "Óbito por outras causa", 9: "Ignorado"
    }

    columns_yes_no = [
        'FEBRE', 'DISPNEIA', 'DESC_RESP',
        'FATOR_RISC', 'VACINA', 'ANTIVIRAL', 'UTI', 'VACINA_COV', 'SURTO_SG',
    ]

    df_treated = df.copy()

    df_treated[columns_yes_no] = df_treated[columns_yes_no].apply(lambda col: col.map(map_yes_no))

    df_treated['AVE_SUINO'] = df_treated['AVE_SUINO'].map(map_birds_pigs)
    df_treated['EVOLUCAO'] = df_treated['EVOLUCAO'].map(map_evolucao)

    df_treated['DT_NOTIFIC'] = pd.to_datetime(df_treated['DT_NOTIFIC'], errors='coerce') \
        .dt.strftime('%Y-%m-%d')

    df_treated.to_csv(output_path, sep=";", index=False)
    logging.info(f"Data saved to: {output_path}")

    return df_treated


if __name__ == "__main__":

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    silver_path = os.path.join(project_root, "datalake", "silver", "srag_treated_2025.csv")

    df_processed = final_processing_data(silver_path)

    logging.info("Final processing done.")
