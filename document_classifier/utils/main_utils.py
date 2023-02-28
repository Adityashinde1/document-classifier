import shutil
import sys
from typing import Dict
import dill
import pickle
import json
import numpy as np
import yaml
from zipfile import Path
from document_classifier.constant import *
from document_classifier.exception import DocumentClassifierException
import logging

# initiatlizing logger
logger = logging.getLogger(__name__)


class MainUtils:
    def read_yaml_file(self, filename: str) -> Dict:
        logger.info("Entered the read_yaml_file method of MainUtils class")
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    @staticmethod
    def dump_pickle_file(output_filepath: str, data) -> None: 
        try:
            with open(output_filepath, "wb") as encoded_pickle:
                pickle.dump(data, encoded_pickle)

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    @staticmethod
    def load_pickle_file(filepath: str) -> object:
        try:
            with open(filepath, "rb") as pickle_obj:
                obj = pickle.load(pickle_obj)
            return obj

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
        

    @staticmethod
    def dump_json_file(data: dict, filepath: str):
        try:
            with open(filepath, "w") as outfile:
                json.dump(data, outfile)

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
        

    @staticmethod
    def read_json_file(filepath: str):
        try:
            with open(filepath, 'r') as openfile:
                json_object = json.load(openfile)

            return json_object
        
        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
            

    def save_numpy_array_data(self, file_path: str, array: np.array) -> str:
        logger.info("Entered the save_numpy_array_data method of MainUtils class")
        try:
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
            logger.info("Exited the save_numpy_array_data method of MainUtils class")
            return file_path

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    def load_numpy_array_data(self, file_path: str) -> np.array:
        logger.info("Entered the load_numpy_array_data method of MainUtils class")
        try:
            with open(file_path, "rb") as file_obj:
                return np.load(file_obj)

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        logger.info("Entered the save_object method of MainUtils class")
        try:
            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)

            logger.info("Exited the save_object method of MainUtils class")

            return file_path

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    @staticmethod
    def load_object(file_path: str) -> object:
        logger.info("Entered the load_object method of MainUtils class")
        try:
            with open(file_path, "rb") as file_obj:
                obj = dill.load(file_obj)
            logger.info("Exited the load_object method of MainUtils class")
            return obj

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    @staticmethod
    def create_artifacts_zip(file_name: str, folder_name: str) -> None:
        logger.info("Entered the create_artifacts_zip method of MainUtils class")
        try:
            shutil.make_archive(file_name, "zip", folder_name)
            logger.info("Exited the create_artifacts_zip method of MainUtils class")

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    @staticmethod
    def unzip_file(filename: str, folder_name: str) -> None:
        logger.info("Entered the unzip_file method of MainUtils class")
        try:
            shutil.unpack_archive(filename, folder_name)
            logger.info("Exited the unzip_file method of MainUtils class")

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    @staticmethod        
    def read_txt_file(filename: str) -> str:
        logger.info("Entered the read_txt_file method of MainUtils class")
        try:
            # Opening file for read only
            file1 = open(filename, 'r')
            # read all text
            text = file1.read()
            # close the file
            file1.close()
            logger.info("Exited the read_txt_file method of MainUtils class")
            return text

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    @staticmethod
    def save_descriptions(descriptions, filename) -> None:
        try:
            lines = list()
            for key, desc_list in descriptions.items():
                for desc in desc_list:
                    lines.append(key + ' ' + desc)
            data = '\n'.join(lines)
            file1 = open(filename, 'w')
            file1.write(data)
            file1.close()
            return filename

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    @staticmethod
    def save_txt_file(output_file_path: str, data: list) -> Path: 
        try:
            with open(output_file_path, "w") as file:
                file.writelines("% s\n" % line for line in data)

            return output_file_path

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    @staticmethod
    def max_length_desc(descriptions: dict) -> int:
        try:
            all_desc = list()
            for key in descriptions.keys():
                [all_desc.append(d) for d in descriptions[key]]
            return max(len(d.split()) for d in all_desc)

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e