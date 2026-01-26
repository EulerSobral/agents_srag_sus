import pandas as pd 
from datetime import datetime, timedelta 
import logging 
import os 
import json  


class MetricsCalculator: 
    """ 
    A class to calculate various metrics from a dataset.
    """

    def __init__(self, path_to_data: str, output_path: str): 
        """ 
        Initialize the MetricsCalculator with data from the specified path.
        """ 

        try: 
            df = pd.read_csv(path_to_data, sep=";") 
            self.df = df
            logging.info(f"Data loaded successfully from {path_to_data}.")  
            df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'], errors='coerce')
            self.save_metrics_to_json(df, output_path) 
           
        except Exception as e: 
            logging.error(f"Error loading data from {path_to_data}: {e}") 
            raise e  


    def _calculate_metrics_increase_date(self, df, current_start_date, current_end_date, previous_start_date, previous_end_date) -> float: 
        """ 
        Calculate the percentage of case increase between two date ranges.
        """   
        current_event_cases = df[(df['DT_NOTIFIC'] >= current_start_date) & (df['DT_NOTIFIC'] <= current_end_date)].shape[0] 
        previous_event_cases = df[(df['DT_NOTIFIC'] >= previous_start_date) & (df['DT_NOTIFIC'] <= previous_end_date)].shape[0] 
        
        if previous_event_cases == 0: 
            return 0 

        return ((current_event_cases - previous_event_cases) / previous_event_cases) * 100

    def _calculate_metrics_ave_suino(self, df, start_date, end_date) -> float:  

        interval_df = df[(df['DT_NOTIFIC'] >= start_date) & (df['DT_NOTIFIC'] <= end_date)]

        ave_suino_evolution = interval_df[interval_df['AVE_SUINO'].isin(["Sim, aves e/ou suínos", "Sim, outros", "Não, nenhum"])]
    
        if ave_suino_evolution.empty: 
            logging.info("No 'AVE_SUINO' cases found in the specified date range.") 
            return 0.0
        
        ave_suino_case = ave_suino_evolution[ave_suino_evolution['AVE_SUINO'].isin(["Sim, aves e/ou suínos", "Sim, outros"])].shape[0]
        percentage = (ave_suino_case / ave_suino_evolution.shape[0]) * 100

        logging.info(f"Percentage of 'AVE_SUINO' cases calculated: {percentage:.2f}%") 
        return percentage
 

    def _calculate_metrics_febre(self, df, start_date, end_date) -> float:  

        interval_df = df[(df['DT_NOTIFIC'] >= start_date) & (df['DT_NOTIFIC'] <= end_date)]

        febre_evolution = interval_df[interval_df['FEBRE'].isin(["Sim", "Não"])]

        if febre_evolution.empty: 
            logging.info("No 'FEBRE' cases found in the specified date range.") 
            return 0.0
        febre_case = febre_evolution[febre_evolution['FEBRE'] == "Sim"].shape[0]
        percentage = (febre_case / febre_evolution.shape[0]) * 100
        logging.info(f"Percentage of 'FEBRE' cases calculated: {percentage:.2f}%") 
        return percentage

    def _calculate_metrics_dispneia(self, df, start_date, end_date) -> float:  

        interval_df = df[(df['DT_NOTIFIC'] >= start_date) & (df['DT_NOTIFIC'] <= end_date)]

        dispneia_evolution = interval_df[interval_df['DISPNEIA'].isin(["Sim", "Não"])]

        if dispneia_evolution.empty: 
            logging.info("No 'DISPNEIA' cases found in the specified date range.") 
            return 0.0
        dispneia_case = dispneia_evolution[dispneia_evolution['DISPNEIA'] == "Sim"].shape[0]
        percentage = (dispneia_case / dispneia_evolution.shape[0]) * 100
        logging.info(f"Percentage of 'DISPNEIA' cases calculated: {percentage:.2f}%") 
        return percentage


    def _calculate_metrics_fator_risc(self, df, start_date, end_date) -> float:   

        interval_df = df[(df['DT_NOTIFIC'] >= start_date) & (df['DT_NOTIFIC'] <= end_date)]

        fator_risc_evolution = interval_df[interval_df['FATOR_RISC'].isin(["Sim", "Não"])]

        if fator_risc_evolution.empty: 
            logging.info("No 'FATOR_RISC' cases found in the specified date range.") 
            return 0.0
        fator_risc_case = fator_risc_evolution[fator_risc_evolution['FATOR_RISC'] == "Sim"].shape[0]
        percentage = (fator_risc_case / fator_risc_evolution.shape[0]) * 100
        logging.info(f"Percentage of 'FATOR_RISC' cases calculated: {percentage:.2f}%") 
        return percentage
  
    def _calculate_metrics_vacina(self, df, start_date, end_date) -> float:   

        interval_df = df[(df['DT_NOTIFIC'] >= start_date) & (df['DT_NOTIFIC'] <= end_date)]

        vacina_evolution = interval_df[interval_df['VACINA'].isin(["Sim", "Não"])]

        if vacina_evolution.empty: 
            logging.info("No 'VACINA' cases found in the specified date range.") 
            return 0.0

        vacina_applied = vacina_evolution[vacina_evolution['VACINA'] == "Sim"].shape[0]
        percentage = (vacina_applied / vacina_evolution.shape[0]) * 100
        logging.info(f"Percentage of applied 'VACINA' calculated: {percentage:.2f}%") 
        return percentage        

    def _calculate_metrics_antiviral(self, df, start_date, end_date) -> float:   

        interval_df = df[(df['DT_NOTIFIC'] >= start_date) & (df['DT_NOTIFIC'] <= end_date)]

        antiviral_evolution = interval_df[interval_df['ANTIVIRAL'].isin(["Sim", "Não"])]

        if antiviral_evolution.empty: 
            logging.info("No 'ANTIVIRAL' cases found in the specified date range.") 
            return 0.0

        antiviral_applied = antiviral_evolution[antiviral_evolution['ANTIVIRAL'] == "Sim"].shape[0]
        percentage = (antiviral_applied / antiviral_evolution.shape[0]) * 100
        logging.info(f"Percentage of applied 'ANTIVIRAL' calculated: {percentage:.2f}%") 
        return percentage       
 
    def _calculate_metrics_uti(self, df, start_date, end_date) -> float:   

        interval_df = df[(df['DT_NOTIFIC'] >= start_date) & (df['DT_NOTIFIC'] <= end_date)]

        uti_evolution = interval_df[interval_df['UTI'].isin(["Sim", "Não"])]

        if uti_evolution.empty: 
            logging.info("No 'UTI' cases found in the specified date range.") 
            return 0.0

        uti_applied = uti_evolution[uti_evolution['UTI'] == "Sim"].shape[0]
        percentage = (uti_applied / uti_evolution.shape[0]) * 100
        logging.info(f"Percentage of applied 'UTI' calculated: {percentage:.2f}%") 
        return percentage           
     
    def _calculate_metrics_evolucao(self, df, start_date, end_date) -> float:   
        interval_df = df[(df['DT_NOTIFIC'] >= start_date) & (df['DT_NOTIFIC'] <= end_date)]

        evolucao_total = interval_df[interval_df['EVOLUCAO'].isin(["Cura", "Óbito"])]

        if evolucao_total.empty: 
            logging.info("No 'EVOLUCAO' cases found in the specified date range.") 
            return 0.0

        evolucao_cases = evolucao_total[evolucao_total['EVOLUCAO'].isin(["Óbito"])].shape[0]
        percentage = (evolucao_cases / evolucao_total.shape[0]) * 100
        logging.info(f"Percentage of 'EVOLUCAO' calculated: {percentage:.2f}%") 
        return percentage 
    
    def _calculate_metrics_surto_sg(self, df, start_date, end_date) -> float:   
        interval_df = df[(df['DT_NOTIFIC'] >= start_date) & (df['DT_NOTIFIC'] <= end_date)]

        surto_sg_total = interval_df[interval_df['SURTO_SG'].isin(["Sim", "Não"])]

        if surto_sg_total.empty: 
            logging.info("No 'SURTO_SG' cases found in the specified date range.") 
            return 0.0

        surto_sg_cases = surto_sg_total[surto_sg_total['SURTO_SG'] == "Sim"].shape[0]
        percentage = (surto_sg_cases / surto_sg_total.shape[0]) * 100
        logging.info(f"Percentage of 'SURTO_SG' calculated: {percentage:.2f}%") 
        return percentage 
    

    def save_metrics_to_json(self, df,output_path: str): 
        """ 
        Save calculated metrics to a JSON file.
        """    
        current_start_date = '2025-01-01'
        current_end_date = '2025-12-31' 
        previous_start_date = '2024-01-01'
        previous_end_date = '2024-12-31' 

        metrics = { 
            "antiviral_increase": self._calculate_metrics_antiviral(df, current_start_date, current_end_date) , 
            "cases_ave_suino": self._calculate_metrics_ave_suino(df, current_start_date, current_end_date), 
            "cases_febre": self._calculate_metrics_febre(df, current_start_date, current_end_date), 
            "cases_dispneia": self._calculate_metrics_dispneia(df, current_start_date, current_end_date), 
            "cases_fator_risc": self._calculate_metrics_fator_risc(df, current_start_date, current_end_date), 
            "cases_vacina": self._calculate_metrics_vacina(df, current_start_date, current_end_date), 
            "cases_uti": self._calculate_metrics_uti(df, current_start_date, current_end_date), 
            "cases_evolucao": self._calculate_metrics_evolucao(df, current_start_date, current_end_date), 
            "cases_surto_sg": self._calculate_metrics_surto_sg(df, current_start_date, current_end_date)
        }   

        with open(output_path, 'w') as f: 
            json.dump(metrics, f, indent=4) 

        logging.info(f"Metrics saved to {output_path}.")  