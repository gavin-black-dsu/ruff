import argparse
import os
import shutil
from tqdm import tqdm, trange
import pandas as pd
import time
import pickle

from statistics_tests import perform_ks, estimate_power_with_data
#from seed_data_mappers import model_dict, file_to_directory, model_dirs
from dataframe_helpers import update_dataframe, extract_prefix, truncate_error
from scoring import normalize_resources, scaled_resources, transformer_embeddings, quantized_vectors, make_anomaly_scores, scale_column
from trial_config import read_config

def main(cur_test, dataframe_loc, config_file='config.ini', iteration=1):
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
        scale_column(df, "Embeddings Distances")
    else:
        df["Embeddings Distances"] = 0
    embedding_time = time.time() 
                
    if gamma_anom > 0: # Skip resource anomalies if not part of score
        df = quantized_vectors(df)
        df = make_anomaly_scores(df, "Resources", k)
        scale_column(df, "Resources Distances")
    else:
        df["Resources Distances"] = 0
    resource_time = time.time()
        
    # No need to skip scaling, it produces minimal overhead
    df = scaled_resources(df)
    scale_column(df, "Scaled Resources")

    scale_time = time.time()

    summed_coefficients = alpha_embed + beta_scaled + gamma_anom
    # Requires the above be run first
    df['Score'] = ( alpha_embed * df['Embeddings Distances'] + 
                    beta_scaled * df['Scaled Resources'] + 
                    gamma_anom * df['Resources Distances']) / summed_coefficients
    
    end_time = time.time()

    print("Avg Scores: ")
    print("  - Embeddings: " + str(df["Embeddings Distances"].max()))
    print("  - Scaled:     " + str(df["Scaled Resources"].max()))
    print("  - Resources:  " + str(df["Resources Distances"].max()))
    print("  - Score:      " + str(df["Score"].max()))

    times = { "Embedding Anomaly Time (ms)": 1000 * (embedding_time - start_time),
              "Resource Anomaly Time (ms)": 1000 * (resource_time - embedding_time),
              "Resource Scaled Time (ms)": 1000 * (scale_time - resource_time),
              "Total Time (ms)": 1000 * (end_time - start_time)
            }
    counts = { "LLM": 0 }

    # Find and step through all the highest scoring samples
    top_samples = df.nlargest(config["number_of_samples"], 'Score')
    count = 0
    score_dir, fuzz_dir = config["score_dir"], config["fuzz_dir"]
    test_name, fuzzer = config["name"], config["fuzzer"]
    for index, row in top_samples.iterrows():
        corpus = row['Corpus Location']
        sample = row['Sample']
        sample_path = f"{corpus}/{sample}"
    
        out_dir = f"{fuzz_dir}/{test_name}/{cur_test}/{fuzzer}/{iteration}/Samples"
          
        count += 1
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(sample_path, f"{out_dir}/{count}.txt")
    
        # Track which corpus each sample comes from
        if "atheris" not in corpus and "random_data" not in corpus:
            counts["LLM"] += 1
        
        if corpus not in counts.keys(): counts[corpus] = 0
        counts[corpus] += 1
    
    # Save the accumulated dataframe and metadata
    df.to_parquet(f"{score_dir}/{test_name}___{cur_test}.parquet")
    with open(f'{score_dir}/{test_name}___{cur_test}.pkl', 'wb') as file:
        pickle.dump({"Top Counts": counts, "Performance": times}, file)
    
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
