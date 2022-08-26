import pandas as pd
import logging

logging.basicConfig(
    filename="./results.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def read_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"SUCCESS:There are {df.shape} rows in your dataframe")
        logging.info(f"SUCCESS:Your file was succesfully read in!")
        return df
    except FileNotFoundError:
        logging.error("ERROR: File not found!!!")


df = read_data("some_path")
