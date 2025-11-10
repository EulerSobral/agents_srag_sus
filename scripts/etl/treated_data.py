from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
        .appName("TreatedData") \
        .master("local[*]") \
        .getOrCreate()

def load_data(file_path):
    df = spark.read.parquet(file_path)
    return df  

def treated_data(df):
    columns_important = [
        'NU_IDADE_N', 'CS_GESTANT', 'AVE_SUINO', 'FEBRE',  
        'TOSSE', 'GARGANTA', 'DISPNEIA', 'DESC_RESP',  
        'DIARREIA', 'VOMITO', 'FATOR_RISC', 
        'VACINA', 'ANTIVIRAL', 'TP_ANTIVIR', 'UTI', 
        'TP_AMOSTRA', 'EVOLUCAO', 'VACINA_COV', 
        'SURTO_SG', 'CO_DETEC'
    ]

    df_raw_important = df.select(*columns_important)
    df_treated = df_raw_important.fillna("0", subset=columns_important)

    path_save = "/home/euler/projeto_srag_agents/datalake/silver" 
 
    
    df_treated.write.format("csv") \
        .option("header", "true") \
        .option("delimiter", ",") \
        .mode("overwrite") \
        .save(path_save)
    
if __name__ == "__main__":  
    file_path = "/home/euler/projeto_srag_agents/datalake/bronze/*.parquet"
    
    df = load_data(file_path) 

    df.printSchema() 
    df.show(5)      
    
    treated_data(df)
    spark.stop()