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
            self.log = CustomLogger().get_logger(__file__)
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

    def save_dataset(self, dataset,filename):
        try:
            
            
            save_path = os.path.join(self.session_path, filename)
            
            dataset.to_csv(save_path,index=False)
            
            self.log.info(
                f"CSV saved successfully",
                filename=filename,
                save_path=save_path,
                session_id=self.session_id
            )
            
            return self.session_id
        except Exception as e:
            self.log.error('Error saving CSV', error=str(e))
            raise AutoML_Exception("Error saving CSV", e) from e




if __name__ == "__main__":
    try:
        df = pd.read_csv(r'D:\College\Project\automl\data\Data_Train.csv')
        handler  = datasetHandler()
        print(handler.save_dataset(df,"raw_file.csv"))
    except AutoML_Exception as e:
        print("AutoML Exception:", e)
    except Exception as e:
        print("Unexpected Error:", e)