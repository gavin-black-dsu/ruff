import os
import string
import pandas as pd
import pyarrow.parquet as pq
from seed_data_mappers import model_dict


def sanitize_content(content):
    # Replace non-printable characters with a placeholder (e.g., '?')
    printable = set(string.printable)
    sanitized = ''.join(filter(lambda x: x in printable, content))
    return sanitized

def load_and_sanitize_file(row):
    directory = row['Corpus']
    filename = row['Sample']
    file_path = os.path.join(directory, filename)

    try:
        with open(file_path, 'rb') as file:
            content = file.read()
            # Decode binary content to string, ignoring errors
            content_str = content.decode('utf-8', errors='ignore')
            sanitized_content = sanitize_content(content_str)
            return sanitized_content
    except Exception as e:
        # Handle errors (e.g., file not found, read errors)
        print(f"Error reading file {file_path}: {e}")
        return None
    
def replace_model_substring(corpus_value):
        for key, value in model_dict.items():
            if key in corpus_value:
                corpus_value = value
        return corpus_value

def truncate_field(v ,num):
    return v[:num]

def truncate_error(df, func_name, error_len=10):
    # Specific fix for ics_Calendar
    def calendar_fix(val):
        if "ALPHADIGIT_MINUS_PLUS" in val: return "ALPHADIGIT_MINUS_PLUS"
        if "Expecting end of text" in val: return "Expecting end of text"
        if val.startswith("(") and val.endswith(")"):
            return val[-2:]
        return val
    
    if func_name == "ics_Calendar": df['Error'] = df['Error'].apply(calendar_fix)
    else: df["Error"] = df['Error'].apply(lambda x: truncate_field(x, error_len))
    return df

def update_dataframe(df):
    # Apply the function to each row and update the 'Sample' column
    df['Sample Text'] = df.apply(load_and_sanitize_file, axis=1)
    #df["Top Function Calls"] = pd.to_numeric(df["Top Function Calls"])
    df['Corpus Location'] = df['Corpus'] # Still keep the original location
    df['Corpus'] = df['Corpus'].apply(lambda x: replace_model_substring(x))

    print(len(df['Sample'].unique()))
    print(len(df['Sample Text'].unique()))
    return df

def extract_prefix(directory):
    prefixes = []
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.parquet'):
                # Split the file name at underscores
                parts = file.split('_')
                if len(parts) > 1:
                    prefix = '_'.join(parts[:-1])
                    prefixes.append(prefix)
                else:
                    prefix = os.path.splitext(file)[0]
                    prefixes.append(prefix)
    return prefixes