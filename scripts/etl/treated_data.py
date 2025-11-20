import pandas as pd  

def load_data(file_path: str): 
    df = pd.read_csv(file_path, sep=";", on_bad_lines="skip")  
    return df

def treated_data(df): 
    
    """ Trata os dados brutos removendo colunas irrelevantes,  
    lidando com valores ausentes e eliminando duplicatas."""
    
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
    file_path = "/home/euler/projeto_srag_agents/datalake/bronze/srag_2025.csv"
    
    df = load_data(file_path) 

    print("Data raw loaded successfully.") 
    print(df.head())
    
    print("Starting data treatment...")
    df_treated = treated_data(df) 
    print(df_treated.head()) 

    print("Saving treated data...") 
    save_treated_data(df_treated, "/home/euler/projeto_srag_agents/datalake/silver/srag_treated_2025.csv")