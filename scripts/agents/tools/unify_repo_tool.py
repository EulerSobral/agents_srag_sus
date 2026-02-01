import os
import json
import base64
import logging



class UnifyRepository:  

        def __init__(self,  
            source_dir_text: str,  
            source_dir_json: str,  
            source_dir_png_last_30_days: str,  
            source_dir_png_last_12_months: str,   
            source_dir_text_delete: str,  
            source_dir_json_delete: str,
            output_file: str):  
            
            try:
                self.source_dir_text = source_dir_text 
                self.source_dir_json = source_dir_json
                self.source_dir_png_last_30_days = source_dir_png_last_30_days
                self.source_dir_png_last_12_months = source_dir_png_last_12_months 
                self.source_dir_text_delete = source_dir_text_delete
                self.source_dir_json_delete = source_dir_json_delete
                self.output_file = output_file     
                 
                content_output_md = self._load_data_file(self.source_dir_text, is_json=False) 
                content_output_json = self._load_data_file(self.source_dir_json, is_json=True)    
                content_output_png_last_30_days = self._load_png_file(self.source_dir_png_last_30_days)
                content_output_png_last_12_months = self._load_png_file(self.source_dir_png_last_12_months)
                self._delete_file(self.source_dir_text_delete)
                self._delete_file(self.source_dir_json_delete)
                self._save_data_in_file(content_output_md + "\n\n" + "\n## Dados das métricas\n\n" + content_output_json + "\n\n" + "\n## Gráficos da evolução nos últimos 30 dias\n\n" + content_output_png_last_30_days + "\n\n" + "\n## Gráficos da evolução nos últimos 12 meses\n\n" + content_output_png_last_12_months, self.output_file)
                logging.info("Initialized UnifyRepository successfully.")

            except Exception as e:
                logging.error(f"Error initializing UnifyRepository: {e}")
                raise e  

        def _load_data_file(self, file_path: str, is_json: bool) -> str:  
            """
            Get the content of a data file, either JSON or plain text.
            """    

            try:
                with open(file_path, 'r') as f:
                    if is_json:
                        data = json.load(f)  
                        logging.info(f"Loaded JSON data from {file_path}")
                        formatted_text = "\n".join(f"{key}: {value}" for key, value in data.items())
                        return formatted_text
                    logging.info(f"Loaded text data from {file_path}")
                    return f.read()  
            except Exception as e:
                logging.error(f"Error loading data from {file_path}: {e}")
                raise e  

        def _save_data_in_file(self, content: str, output_file: str):
            """
            Save the unified content into the output markdown file.
            """    
            try:
                with open(output_file, 'w') as f:
                    f.write(content)
                logging.info(f"Unified content saved to {output_file}")
            except Exception as e:
                logging.error(f"Error saving unified content to {output_file}: {e}")
                raise e     

        def _delete_file(self, file_path: str):
            """
            Delete a file if it exists.
            """
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Deleted file: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")
                raise e

         

        def _load_png_file(self, png_path: str) -> str:
            """
            Load a PNG file and return it as a Markdown image link.
            """
            try:
                rel_path = os.path.basename(png_path)
                return f"![Imagem: {rel_path}]({rel_path})\n\n"
            except Exception as e:
                logging.error(f"Error referencing PNG from {png_path}: {e}")
                return f"Erro ao referenciar imagem: {e}\n\n"  