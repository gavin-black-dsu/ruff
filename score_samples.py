import configparser
import argparse
import os
import shutil
from tqdm import tqdm, trange
import pandas as pd
import time

from statistics_tests import perform_ks, estimate_power_with_data
from seed_data_mappers import model_dict, file_to_directory, model_dirs
from dataframe_helpers import update_dataframe, extract_prefix, truncate_error
from scoring import normalize_resources, scaled_resources, transformer_embeddings, quantized_vectors, make_anomaly_scores

def read_config(config_file):
    """
    Reads the configuration from the given INI file and returns a dictionary
    of configurations.
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    print(config_file)
    settings = {
        'name': config.get('Trial', 'name'),
        'directory': config.get('Paths', 'directory'),
        'corpora_dir': config.get('Paths', 'corpora_dir'),
        'fuzz_dir': config.get('Paths', 'fuzz_dir'),
        'max_errors': config.getint('Settings', 'max_errors'),
        'n_simulations': config.getint('Settings', 'n_simulations'),
        'ks_a': config.getfloat('Settings', 'ks_a'),
        'k': config.getint('Settings', 'k'),
        'limit_atheris': config.getboolean('Settings', 'limit_atheris'),
        'atheris_entries': config.getint('Settings', 'atheris_entries'),
        'batch_size': config.getint('Settings', 'batch_size'),
        'number_of_samples': config.getint('Fuzzing', 'number_of_samples'),
        'fuzzer': config.get('Fuzzing', 'fuzzer'),

        'alpha_embed': config.getfloat('Score Coefficients', 'alpha_embed'),
        'beta_scaled': config.getfloat('Score Coefficients', 'beta_scaled'),
        'gamma_anom': config.getfloat('Score Coefficients', 'gamma_anom'),
        'scaling_factors': {
            'Output_Size_B': config.getfloat('Scaling', 'Output_Size_B'),
            'Output_Entropy_bits_per_symbol': config.getfloat('Scaling', 'Output_Entropy_bits_per_symbol'),
            'Top_Function_Calls': config.getfloat('Scaling', 'Top_Function_Calls'),
            'Peak_Memory_B': config.getfloat('Scaling', 'Peak_Memory_B'),
            'Final_Memory_B': config.getfloat('Scaling', 'Final_Memory_B'),
            'CPU_ms': config.getfloat('Scaling', 'CPU_ms'),
            'SLOC': config.getfloat('Scaling', 'SLOC'),
            'Instructions': config.getfloat('Scaling', 'Instructions'),
        }
    }

    return settings

def main(cur_test, dataframe_loc, config_file='config.ini', iteration=1) :
    # Parse the configuration file
    config = read_config(config_file)

    k = config["k"] # Number of anomaly distances
    alpha_embed, beta_scaled, gamma_anom = config["alpha_embed"], config["beta_scaled"], config["gamma_anom"]
        
    df = pd.read_parquet(dataframe_loc)
    df = df.sample(frac=1).reset_index(drop=True)

    if config["limit_atheris"]:
        atheris_df = df[df['Corpus'] == 'Atheris']
        other_df = df[df['Corpus'] != 'Atheris']
        atheris_df = atheris_df.sample(n=config["atheris_entries"], random_state=1) 
        df = pd.concat([atheris_df, other_df], ignore_index=True)
        
    start_time = time.time()
        
    # Apply Scoring
    if alpha_embed > 0: # Skip the embeddings if not part of score
        df = transformer_embeddings(df, batch_size = config["batch_size"])
        df = make_anomaly_scores(df, "Embeddings", k)
    else:
        df["Embeddings Distances"] = 0
    embedding_time = time.time() 
        
        
    if gamma_anom > 0: # Skip resource anomalies if not part of score
        df = quantized_vectors(df)
        df = make_anomaly_scores(df, "Resources", k)
    else:
        df["Resources Distances"] = 0
    resource_time = time.time()
        
    # No need to skip scaling, it produces minimal overhead
    df = scaled_resources(df)
    scale_time = time.time()

    summed_coefficients = alpha_embed + beta_scaled + gamma_anom
    # Requires the above be run first
    df['Score'] = ( alpha_embed * df['Embeddings Distances'] + 
                    beta_scaled * df['Scaled Resources'] + 
                    gamma_anom * df['Resources Distances']) / summed_coefficients
    
    end_time = time.time()

    print("Max Scores: ")
    v = df["Embeddings Distances"].max()
    print(f"  - Embeddings: {v}")
    v = df["Scaled Resources"].max()
    print(f"  - Scaled:     {v}")
    v = df["Resources Distances"].max()
    print(f"  - Resources:  {v}")
    v = df["Score"].max()
    print(f"  - Score:      {v}")

    times = { "Embedding Anomaly Time (ms)": 1000 * (embedding_time - start_time),
              "Resource Anomaly Time (ms)": 1000 * (resource_time - embedding_time),
              "Resource Scaled Time (ms)": 1000 * (scale_time - resource_time),
              "Total Time (ms)": 1000 * (end_time - start_time)
            }
    
    top_samples = df.nlargest(config["number_of_samples"], 'Score')
    count = 0
    for index, row in top_samples.iterrows():
        corpus = row['Corpus']
        sample = row['Sample']

        corpora_dir, fuzz_dir = config["corpora_dir"], config["fuzz_dir"]
        test_name, fuzzer = config["name"], config["fuzzer"]
        c_dir = file_to_directory[f"file_{cur_test}.py"]
        merge = "/merge_corpus"
    
        if corpus == "Atheris" or corpus == "Random": merge = ""
        sample_path = f"{corpora_dir}/{c_dir}/{model_dirs[corpus]}{merge}/{sample}"
        
     
        out_dir = f"{fuzz_dir}/{test_name}/{cur_test}/{fuzzer}/{iteration}/Samples"
        count += 1
        os.makedirs(out_dir, exist_ok=True)
        #print(sample_path, f"{out_dir}/{count}.txt")
        shutil.copy(sample_path, f"{out_dir}/{count}.txt")

if __name__ == '__main__':
    # Setup argparse for command line arguments
    parser = argparse.ArgumentParser(description='Read configuration from INI file and analyze resources.')
    parser.add_argument('-c', '--config', type=str, default='config.ini',
                        help='Path to the configuration file (default: config.ini)')
    parser.add_argument('-i', '--iteration', type=int, default=1,
                        help='Number for tracking the fuzzing iteration')
    parser.add_argument('function_name', type=str, help='Name of the function to analyze')
    parser.add_argument('resources_dataframe', type=str, help='Path to the parquet file containing resources data')
    args = parser.parse_args()

    # Call main function with the provided arguments
    main(args.function_name, args.resources_dataframe, args.config, args.iteration)
