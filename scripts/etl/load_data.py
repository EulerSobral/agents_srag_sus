import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType, StringType # Importe os tipos

spark = SparkSession.builder \
        .appName("BeatterTreatedData") \
        .master("local[*]") \
        .getOrCreate() 

def final_processing_data(file_path):
    print(f"Iniciando processamento. Lendo arquivos de: {file_path}")
    
    try:
        df = spark.read.csv(
            file_path, 
            header=True, 
            sep=",", 
        ) 
        print(df.printSchema())
    except Exception as e:
        print(f"Erro ao ler os arquivos CSV. Verifique o caminho e o esquema. Erro: {e}")
        return

    map_yes_no = { 
        "0": "Campo vazio",
        "1": "Sim",
        "2": "Não",
        "9": "Ignorado"
    }

    map_birds_pigs = { 
        "0": "Campo vazio",
        "1": "Sim, aves e/ou suínos",
        "2": "Não, nenhum",
        "3": "Sim, outros",
        "9": "Ignorado"
    }

    map_pregnant = { 
        "0": "Campo vazio",
        "1": "1º Trimestre",
        "2": "2º Trimestre",
        "3": "3º Trimestre",
        "4": "Idade Gestacional Ignorada",
        "5": "Não",
        "6": "Não se aplica",
        "9": "Ignorado"
    }

    map_garganta = { 
        "0": "Campo vazio",
        "1": "Sim",
        "2": "Não"
    }

    map_tp_antivir = { 
        "0": "Campo vazio",
        "1": "Oseltamivir",
        "2": "Zanamivir",
        "3": "Outro"
    }

    map_tipo_amostra = { 
        "0": "Campo vazio",
        "1": "Secreção de Naso-orofaringe",
        "2": "Lavado Bronco-alveolar",
        "3": "Tecido post-mortem",
        "4": "Outra",
        "5": "LCR",
        "9": "Ignorado"
    }

    map_evolucao = { 
        "0": "Campo vazio",
        "1": "Cura",
        "2": "Óbito",
        "3": "Óbito por outras causa",
        "9": "Ignorado"
    }

    columns_yes_no = [
        'FEBRE', 'TOSSE', 'DISPNEIA', 'DESC_RESP', 'DIARREIA', 'VOMITO', 
        'FATOR_RISC', 'VACINA', 'ANTIVIRAL', 'UTI', 'VACINA_COV', 'SURTO_SG', 
        'CO_DETEC'
    ] 

    df_treated = df.replace(map_yes_no, subset=columns_yes_no) \
                   .replace(map_birds_pigs, subset=['AVE_SUINO']) \
                   .replace(map_pregnant, subset=['CS_GESTANT']) \
                   .replace(map_garganta, subset=['GARGANTA']) \
                   .replace(map_tp_antivir, subset=['TP_ANTIVIR']) \
                   .replace(map_tipo_amostra, subset=['TP_AMOSTRA']) \
                   .replace(map_evolucao, subset=['EVOLUCAO'])

    path_save = "/home/euler/projeto_srag_agents/datalake/gold" 
    print(f"Salvando dados tratados em: {path_save}")
    df_treated.write.format("csv").option("header", "true").option("delimiter", ",").mode("overwrite").save(path_save)
    print("Dados salvos com sucesso.")

if __name__ == "__main__":
    file_path = "/home/euler/projeto_srag_agents/datalake/silver/*.csv"
    
    final_processing_data(file_path)
    
    print("Processamento concluído. Parando a sessão Spark.")
    spark.stop()