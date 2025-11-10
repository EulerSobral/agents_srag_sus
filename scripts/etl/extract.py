import requests   


def extract_data() -> None: 

    for i in range(2019, 2025):   
        j = str(i)[-2:]
        url = f"https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/{i}/INFLUD{j}-26-06-2025.parquet"
        output_path = f"/home/euler/projeto_srag_agents/datalake/bronze/srag_{i}.parquet"
        
        response = requests.get(url)
        response.raise_for_status()  
        
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"Data for year {i} extracted and saved to {output_path}")    
if __name__ == "__main__":
    extract_data()