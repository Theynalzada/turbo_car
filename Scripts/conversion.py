# Importing Dependencies
import pandas as pd
import warnings

# Filtering potential warnings
warnings.filterwarnings(action = "ignore")

# Defining the target path
target_path = '/Users/kzeynalzade/Documents/Turbo Project/Data/Raw Data/raw_data.csv'

# Defining a function to convert a csv file to a parquet file
def convert_data(filepath = None):
    """
    This is a function to convert a csv file to a parquet file.
    
    Args:
        filepath: A file path to a dataset.
        
    Returns:
        Writes a dataset to a separate parquet file.
    """
    # Reading the data from a csv file
    dataset = pd.read_csv(filepath_or_buffer = filepath)
    
    # Defining the directory path
    directory_path = '/Users/kzeynalzade/Documents/Turbo Project/Data/Converted Data'
    
    # Writing the dataset to a separate parquet file
    dataset.to_parquet(path = f"{directory_path}/input_data.parquet.brotli", 
                       engine = "fastparquet", 
                       compression = "brotli", 
                       index = False)
    
# Running the script
if __name__ == "__main__":
    # Calling the function
    convert_data(filepath = target_path)