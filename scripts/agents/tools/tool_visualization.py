import os
import pandas as pd 
import matplotlib.pyplot as plt 

def visualize_last_30_days(df, output_path): 
    df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC']) 

    end_date = df['DT_NOTIFIC'].max() 
    start_date = end_date - pd.Timedelta(days=30) 

    filter = (df['DT_NOTIFIC'] >= start_date) & (df['DT_NOTIFIC'] <= end_date) 

    df_last_30_days = df['DT_NOTIFIC'][filter].value_counts() 

    plt.figure(figsize=(15,8)) 

    plt.title("Número de casos nos últimos 30 dias") 
    df_last_30_days.plot(kind='bar')
    plt.xlabel("Data")
    plt.ylabel("Quantidade") 

    plt.savefig(output_path)   
    plt.close()