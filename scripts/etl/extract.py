import requests   

YEAR = 2025 
FINAL_NUMBER_YEAR = 25

def extract_data() -> None: 

    url = f"https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2025/INFLUD25-10-11-2025.csv"
    output_path = f"/home/euler/projeto_srag_agents/datalake/bronze/srag_{YEAR}.csv"
        
    response = requests.get(url)
    response.raise_for_status()  
        
    with open(output_path, 'wb') as file:
        file.write(response.content)
    print(f"Data for year {YEAR} extracted and saved to {output_path}")    
if __name__ == "__main__":
    extract_data()