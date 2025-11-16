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

def visualize_last_12_months(df, output_path): 
    df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'])

    end_date = df['DT_NOTIFIC'].max()
    start_date = end_date - pd.DateOffset(months=12)

    filter = (df['DT_NOTIFIC'] >= start_date) & (df['DT_NOTIFIC'] <= end_date)


    df_last_12_months = df[filter]  

    monthly_counts = df_last_12_months.groupby( 
        df_last_12_months['DT_NOTIFIC'].dt.strftime('%Y-%m')
    ).size()

    plt.figure(figsize=(15,8))

    plt.title("Número de casos nos últimos 12 meses")
    monthly_counts.plot(kind='bar')
    plt.xlabel("Data")
    plt.ylabel("Quantidade") 

    plt.savefig(output_path) 
    plt.close()