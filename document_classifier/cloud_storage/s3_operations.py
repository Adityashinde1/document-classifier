import logging
import os
import pickle
import sys
from io import StringIO
from typing import List, Union

import boto3
from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import Bucket
from pandas import DataFrame, read_csv

from document_classifier.constant import *
from document_classifier.exception import DocumentClassifierException

# initializing logger
logger = logging.getLogger(__name__)


class S3Operation:
    def __init__(self):
        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3")

    @staticmethod
    def read_object(
        object_name: str, decode: bool = True, make_readable: bool = False
    ) -> Union[StringIO, str]:

        """
        Method Name :   read_object

        Description :   This method reads the object_name object with kwargs
        
        Output      :   The column name is renamed 
        """
        logger.info("Entered the read_object method of S3Operations class")
        try:
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode is True
                else object_name.get()["Body"].read()
            )

            conv_func = lambda: StringIO(func()) if make_readable is True else func()
            logger.info("Exited the read_object method of S3Operations class")
            return conv_func()

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e

    def get_bucket(self, bucket_name: str) -> Bucket:

        """
        Method Name :   get_bucket

        Description :   This method gets the bucket object based on the bucket_name
        
        Output      :   Bucket object is returned based on the bucket name
        """
        logger.info("Entered the get_bucket method of S3Operations class")
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logger.info("Exited the get_bucket method of S3Operations class")
            return bucket

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    def get_image(self, bucket_name: str, prefix: str, total_number_of_images: int) -> List:

        """
        Method Name :   get_image

        Description :   This method gets the images from the mentioned prefix(folder name)
        
        Output      :   Bucket object is returned based on the bucket name and prefix
        """
        logger.info("Entered the get_image method of S3Operations class")
        try:
            my_bucket = self.s3_resource.Bucket(bucket_name)
            bucket_list = []
            for file in my_bucket.objects.filter(Prefix = prefix):
                file_name = file.key
                if file_name.find(".jpg") != -1:
                    bucket_list.append(file.key)
            
            logger.info("Exited the get_image method of S3Operations class")
            return bucket_list[0:total_number_of_images]

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e

    def is_model_present(self, bucket_name: str, s3_model_key: str) -> bool:

        """
        Method Name :   is_model_present

        Description :   This method validates whether model is present in the s3 bucket or not.
        
        Output      :   True or False
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [
                file_object
                for file_object in bucket.objects.filter(Prefix=s3_model_key)
            ]
            if len(file_objects) > 0:
                return True
            else:
                return False

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e

    def get_file_object(
        self, filename: str, bucket_name: str
    ) -> Union[List[object], object]:
        """
        Method Name :   get_file_object

        Description :   This method gets the file object from bucket_name bucket based on filename 
        
        Output      :   list of objects or object is returned based on filename

        """
        logger.info("Entered the get_file_object method of S3Operations class")
        try:
            bucket = self.get_bucket(bucket_name)
            lst_objs = [object for object in bucket.objects.filter(Prefix=filename)]
            func = lambda x: x[0] if len(x) == 1 else x
            file_objs = func(lst_objs)
            logger.info("Exited the get_file_object method of S3Operations class")
            return file_objs

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e

    def load_model(
        self, model_name: str, bucket_name: str, model_dir: str = None
    ) -> object:

        """
        Method Name :   load_model

        Description :   This method loads the model_name from bucket_name bucket with kwargs
        
        Output      :   list of objects or object is returned based on filename
        """
        logger.info("Entered the load_model method of S3Operations class")

        try:
            func = (
                lambda: model_name
                if model_dir is None
                else model_dir + "/" + model_name
            )
            model_file = func()
            f_obj = self.get_file_object(model_file, bucket_name)
            model_obj = self.read_object(f_obj, decode=False)
            model = pickle.loads(model_obj)
            logger.info("Exited the load_model method of S3Operations class")
            return model

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e

    def create_folder(self, folder_name: str, bucket_name: str) -> None:

        """
        Method Name :   create_folder

        Description :   This method creates a folder_name folder in bucket_name bucket
        
        Output      :   Folder is created in s3 bucket
        """
        logger.info("Entered the create_folder method of S3Operations class")

        try:
            self.s3_resource.Object(bucket_name, folder_name).load()

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                folder_obj = folder_name + "/"
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)
            else:
                pass
            logger.info("Exited the create_folder method of S3Operations class")

    def upload_file(
        self,
        from_filename: str,
        to_filename: str,
        bucket_name: str,
        remove: bool = True,
    ) -> None:

        """
        Method Name :   upload_file

        Description :   This method uploads the from_filename file to bucket_name bucket with to_filename as bucket filename
        
        Output      :   Folder is created in s3 bucket
        """
        logger.info("Entered the upload_file method of S3Operations class")
        try:
            logger.info(
                f"Uploading {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            self.s3_resource.meta.client.upload_file(
                from_filename, bucket_name, to_filename
            )
            logger.info(
                f"Uploaded {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            if remove is True:
                os.remove(from_filename)
                logger.info(f"Remove is set to {remove}, deleted the file")
            else:
                logger.info(f"Remove is set to {remove}, not deleted the file")
            logger.info("Exited the upload_file method of S3Operations class")

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e

    def upload_folder(self, folder_name: str, bucket_name: str) -> None:

        """
        Method Name :   upload_file

        Description :   This method uploads the from_filename file to bucket_name bucket with to_filename as bucket filename
        
        Output      :   Folder is created in s3 bucket
        """
        logger.info("Entered the upload_folder method of S3Operations class")
        try:
            lst = os.listdir(folder_name)
            for f in lst:
                local_f = os.path.join(folder_name, f)
                dest_f = f
                self.upload_file(local_f, dest_f, bucket_name, remove=False)
            logger.info("Exited the upload_folder method of S3Operations class")

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e

    def upload_df_as_csv(
        self,
        data_frame: DataFrame,
        local_filename: str,
        bucket_filename: str,
        bucket_name: str,
    ) -> None:

        """
        Method Name :   upload_df_as_csv

        Description :   This method uploads the dataframe to bucket_filename csv file in bucket_name bucket 
        
        Output      :   Folder is created in s3 bucket
        """
        logger.info("Entered the upload_df_as_csv method of S3Operations class")
        try:
            data_frame.to_csv(local_filename, index=None, header=True)
            self.upload_file(local_filename, bucket_filename, bucket_name)
            logger.info("Exited the upload_df_as_csv method of S3Operations class")

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e

    def get_df_from_object(self, object_: object) -> DataFrame:

        """
        Method Name :   get_df_from_object

        Description :   This method gets the dataframe from the object_name object 
        
        Output      :   Folder is created in s3 bucket
        """
        logger.info("Entered the get_df_from_object method of S3Operations class")

        try:
            content = self.read_object(object_, make_readable=True)
            df = read_csv(content, na_values="na")
            logger.info("Exited the get_df_from_object method of S3Operations class")
            return df

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:

        """
        Method Name :   get_df_from_object

        Description :   This method gets the dataframe from the object_name object 
        
        Output      :   Folder is created in s3 bucket

        """
        logger.info("Entered the read_csv method of S3Operations class")
        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            df = self.get_df_from_object(csv_obj)
            logger.info("Exited the read_csv method of S3Operations class")
            return df

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    def download_file(self, bucket_name: str, output_file_path: str, key: str) -> None:

        """
        Method Name :   download_file

        Description :   This method downloads the file from the s3 bucket and saves the file in directory 
        
        Output      :   File is saved in local

        """
        logger.info("Entered the download_file method of S3Operation class")
        try:
            self.s3_resource.Bucket(bucket_name).download_file(key, output_file_path)
            logger.info("Exited the download_file method of S3Operation class")

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e


    def read_data_from_s3(self, bucket_filename: str, bucket_name: str, output_filepath: str) -> None:

        """
        Method Name :   read_data_from_s3

        Description :   This method downloads the file from the s3 bucket and saves the file in directory 
        
        Output      :   returns object.

        """
        logger.info("Entered the read_data_from_s3 method of S3Operation class")
        try:
            bucket = self.get_bucket(bucket_name)
            obj = bucket.download_file(Key=bucket_filename, Filename=output_filepath)
            logger.info("Exited the read_data_from_s3 method of S3Operation class")
            return obj
            
        except Exception as e:
            raise DocumentClassifierException(e, sys) from e
        

    def sync_folder_from_s3(
        self, folder: str, bucket_name: str, bucket_folder_name: str
    ) -> None:
        logger.info("Entered the sync_folder_from_s3 method of S3Operation class")
        try:
            command: str = (
                f"aws s3 sync s3://{bucket_name}/{bucket_folder_name}/ {folder} "
            )

            os.system(command)
            logger.info("Exited the sync_folder_from_s3 method of S3Operation class")

        except Exception as e:
            raise DocumentClassifierException(e, sys) from e