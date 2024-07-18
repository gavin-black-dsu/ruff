from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import faiss
import torch

default_scaling = {
       'Output Size (B)': 1,
       'Output Entropy (bits per symbol)' : 1,
       'Top Function Calls': 1, 
       'Peak Memory (B)': 1,
       'Final Memory (B)': 1,
       'CPU (ms)': 1, 
       'SLOC': 1,
       'Instructions': 1, 
}

numerical_columns = [
    "Output Size (B)", "Output Entropy (bits per symbol)", "Top Function Calls",
    "Peak Memory (B)", "Final Memory (B)", "CPU (ms)", "SLOC", "Instructions"
]

def normalize_resources(df):
    scaler = MinMaxScaler()
    df_numerical_scaled = pd.DataFrame( 
        scaler.fit_transform(df[numerical_columns]), 
        columns=[col + " Normalized" for col in numerical_columns]
    )
    df_normalized = pd.concat([df, df_numerical_scaled], axis=1)
    return df_normalized

def scaled_resources(df, scaling=None):
    if scaling is None: scaling = default_scaling
    normalized_columns = [col for col in df.columns if col.endswith('Normalized')]
    total_scaling = sum(scaling[col[:-11]] for col in normalized_columns)  # Removing ' Normalized' part
    df['Scaled Resources'] = df.apply(
        lambda row: sum(row[col] * scaling[col[:-11]] for col in normalized_columns) / total_scaling,
        axis=1
    )
    return df

def transformer_embeddings(df, device="cuda", batch_size=1024):
    print("Loading Model")
    checkpoint = "Salesforce/codet5p-110m-embedding"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
    
    print("Running Batches")
    sample_text = df["Sample Text"].tolist()
    all_results = []

    for i in range(0, len(sample_text), batch_size):
        batch = sample_text[i:i + batch_size]
        inputs = tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad(): 
            embedding = model(**inputs)
        all_results.extend(embedding.cpu().numpy())

    df['Embeddings'] = all_results
    torch.cuda.empty_cache()
    return df

def get_faiss_index(vectors):    
    dimension = len(vectors[0])
    emb = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(emb)
    res = faiss.StandardGpuResources()

    cpu_index = faiss.IndexFlatIP(dimension)
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    index.add(emb)
    return index
    
def get_faiss_distances(index, vectors, k=100):
    emb = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(emb)

    distances, _ = index.search(emb, k) 
    std_distances = np.std(distances, axis=1)
    average_distances = np.mean(distances, axis=1)
    average_distances = np.clip(average_distances, None, 1)
    
    return { "Averages": (1 - average_distances),
             "StdDevs": std_distances
           }

def quantized_vectors(df, include_categories=True):
    # Quantize numeric vectors
    def quantize_column_float(series):
        return pd.cut(series, bins=5, labels=False)
    
    df_quantized = df[numerical_columns].apply(quantize_column_float)
    df_quantized.columns = [col + " Quantized" for col in df_quantized.columns]

    # Combine the quantized data with the normalized and original data
    df = pd.concat([df, df_quantized], axis=1)
    
    # Add categorical values
    label_encoder_error = LabelEncoder()
    label_encoder_function = LabelEncoder()

    df["Error Categorical"] = label_encoder_error.fit_transform(df["Error"])
    df["Top Function Categorical"] = label_encoder_function.fit_transform(df["Top Function"])
    
    columns_to_collect = ["Error Categorical", "Top Function Categorical"]
    if not include_categories: columns_to_collect = []
    for x in numerical_columns: columns_to_collect.append(x)
    
    df['Resources'] = df[columns_to_collect].values.tolist()
    # Return back a combined set
    # "Resources"
    #resource_vectors = [ x + y for x,y in zip(dictionary["Categories"], dictionary["Quantized"])]
    return df

def make_anomaly_scores(df, column, k=100):
    lst = df[column].tolist()
    embed_index = get_faiss_index(lst)
    d = get_faiss_distances(embed_index, lst, k=k)
    df[f"{column} Distances"] = d["Averages"]
    df[f"{column} StdDev"] = d["StdDevs"]
    
    return df

