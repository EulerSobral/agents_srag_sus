import pandas as pd

def final_processing_data(file_path): 

    """ Retira os valores numéricos e converte para valores categóricos mais descritivos. 
    Também converte a coluna de data para o formato YYYY-MM-DD.
    """


    print(f"Iniciando processamento. Lendo arquivos de: {file_path}")
    
    output_path = "/home/euler/projeto_srag_agents/datalake/gold/srag_2025_final_processed.csv"

    try:
        df = pd.read_csv(file_path, sep=";", on_bad_lines="skip") 
        print("Arquivos CSV lidos com sucesso.")
    except Exception as e:
        print(f"Erro ao ler os arquivos CSV. Verifique o caminho e o esquema. Erro: {e}")
        return

    map_yes_no = { 
        0: "Campo vazio", 1: "Sim", 2: "Não", 9: "Ignorado"
    }   
    map_birds_pigs = { 
    0: "Campo vazio", 1: "Sim, aves e/ou suínos", 2: "Não, nenhum", 
        3: "Sim, outros", 9: "Ignorado"
    }
    map_garganta = { 0: "Campo vazio", 1: "Sim", 2: "Não" }
    map_tp_antivir = { 0: "Campo vazio", 1: "Oseltamivir", 2: "Zanamivir", 3: "Outro" }
    map_tipo_amostra = { 
        0: "Campo vazio", 1: "Secreção de Naso-orofaringe", 2: "Lavado Bronco-alveolar",
        3: "Tecido post-mortem", 4: "Outra", 5: "LCR", 9: "Ignorado"
    }
    map_evolucao = { 
        0: "Campo vazio", 1: "Cura", 2: "Óbito", 3: "Óbito por outras causa", 9: "Ignorado"
    }
    columns_yes_no = [
        'FEBRE', 'TOSSE', 'DISPNEIA', 'DESC_RESP', 'DIARREIA', 'VOMITO', 
        'FATOR_RISC', 'VACINA', 'ANTIVIRAL', 'UTI', 'VACINA_COV', 'SURTO_SG', 
        'CO_DETEC'
    ] 

    df_treated = df.copy()

    df_treated[columns_yes_no] = df_treated[columns_yes_no].apply(lambda col: col.map(map_yes_no))

    df_treated['AVE_SUINO'] = df_treated['AVE_SUINO'].map(map_birds_pigs)
    df_treated['GARGANTA'] = df_treated['GARGANTA'].map(map_garganta)
    df_treated['TP_ANTIVIR'] = df_treated['TP_ANTIVIR'].map(map_tp_antivir)
    df_treated['TP_AMOSTRA'] = df_treated['TP_AMOSTRA'].map(map_tipo_amostra)
    df_treated['EVOLUCAO'] = df_treated['EVOLUCAO'].map(map_evolucao)
    df_treated['DT_NOTIFIC'] = pd.to_datetime(df_treated['DT_NOTIFIC'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    df_treated.to_csv(output_path, sep=";", index=False)
    print(f"Dados processados salvos em: {output_path}")
    
    return df_treated

if __name__ == "__main__":
    file_path = "/home/euler/projeto_srag_agents/datalake/silver/srag_treated_2025.csv"
    
    df_processed = final_processing_data(file_path) 

    print("Processamento final concluído.") 
    print(df_processed.head())
    