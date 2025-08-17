import pandas as pd 
import numpy as np 
import os 
import sys
import uuid
from datetime import datetime
from pathlib import Path
from logger.customlogger import CustomLogger
from expection.customExpection import AutoML_Exception
from io import BytesIO
from typing import List




class datasetHandler:   
    def __init__(self, data_dir=None, session_id=None):
        try:
            self.log = CustomLogger().get_logger('Mayuresh')
            self.data_dir = data_dir or os.getenv(
                'DATA_STORAGE_PATH', 
                os.path.join(os.getcwd(), 'data', 'datasetAnalysis')
            )
            self.session_id = session_id or f"session_id_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.session_path = os.path.join(self.data_dir, self.session_id)
            os.makedirs(self.session_path, exist_ok=True)
            
            self.log.info(
                'Dataset Handler initialized',
                session_id=self.session_id,
                session_path=self.session_path
            )

        except Exception as e:
            self.log.error('Error initializing Dataset Handler', error=str(e))
            raise AutoML_Exception("Error initializing Dataset Handler", e) from e

    def save_dataset(self, uploaded_file):
        try:
            filename = os.path.basename(uploaded_file.name)
            
            if not filename.lower().endswith(".csv"):
                raise AutoML_Exception("Invalid file type. Only CSV files are allowed")
            
            save_path = os.path.join(self.session_path, filename)
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            
            df = pd.read_csv(save_path, encoding='utf-8')
            
            self.log.info(
                f"CSV saved successfully",
                filename=filename,
                save_path=save_path,
                session_id=self.session_id
            )
            
            return df
        except Exception as e:
            self.log.error('Error saving CSV', error=str(e))
            raise AutoML_Exception("Error saving CSV", e) from e

class UploadedFile:
    def __init__(self, path):
        self._f = open(path, "rb")
        self.name = os.path.basename(path)
    def read(self):
        return self._f.read()
    def close(self):
        self._f.close()




if __name__ == "__main__":
    try:
        handler = datasetHandler()
        file_path = r"D:\College\Project\automl\data\Data_Train.csv" 
        uploaded_file = UploadedFile(file_path)  
        df = handler.save_dataset(uploaded_file)
        uploaded_file.close() 

        print("Dataset Loaded Successfully:")
        print(df.head())

    except AutoML_Exception as e:
        print("AutoML Exception:", e)
    except Exception as e:
        print("Unexpected Error:", e)