import os
import pandas as pd
import torch
from esm import pretrained
from Bio import SeqIO
import esm
from torch.utils.data import DataLoader, TensorDataset

# Set cache path to a relative directory
cache_dir = os.path.join(os.path.dirname(__file__), "torch_cache")
os.environ['TORCH_HOME'] = cache_dir

# Load pre-trained ESM-2 model
def load_model(device):
    model, alphabet = pretrained.esm2_t12_35M_UR50D()
    model = model.to(device)  # Move model to GPU
    model.eval()  # Set to evaluation mode
    return model, alphabet

# Clean sequence by replacing 'O' with 'X'
def clean_sequence(sequence):
    return sequence.replace('O', 'X')

# Parse FASTA format file
def parse_fasta(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    protein_name = ""
    sequence = ""

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if protein_name:
                data.append((protein_name, clean_sequence(sequence)))  # Clean sequence
            protein_name = line[1:]
            sequence = ""
        else:
            sequence += line

    if protein_name:
        data.append((protein_name, clean_sequence(sequence)))  # Clean last sequence

    return data

# Extract protein sequence features
def extract_features(fasta_file, model, alphabet, device, batch_size=8):
    sequences = parse_fasta(fasta_file)
    batch_converter = alphabet.get_batch_converter()
    batch_output = batch_converter(sequences)
    batch_labels, batch_strs, batch_tokens = batch_output

    batch_tokens = batch_tokens.to(device)

    dataset = TensorDataset(batch_tokens)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    all_embeddings = []
    for batch in data_loader:
        batch_token_sub = batch[0]
        with torch.no_grad():
            results = model(batch_token_sub, repr_layers=[12])
        embeddings = results["representations"][12]
        embeddings = embeddings.mean(dim=1)
        all_embeddings.append(embeddings)
        torch.cuda.empty_cache()

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

# Save features to CSV file
def save_features_as_csv(embeddings, output_file):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    embeddings_np = embeddings.cpu().numpy()
    df = pd.DataFrame(embeddings_np)
    df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

# Merge positive and negative CSV files with labels
def merge_csv_with_labels(pos_file, neg_file, output_file):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Read positive and negative CSV files
    pos_df = pd.read_csv(pos_file)
    neg_df = pd.read_csv(neg_file)

    # Add Label column (1 for positive, 0 for negative)
    pos_df['Label'] = 1
    neg_df['Label'] = 0

    # Concatenate the dataframes
    merged_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # Save the merged dataframe
    merged_df.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file}")

# Main function
def main():
    '''Train'''
    train_pos_fasta_file = "../Data/90train_positive3572.txt"  # Positive FASTA path
    train_pos_output_file = "3572_pos_protein_embeddings_forval.csv"  # Positive output CSV
    train_neg_fasta_file = "../Data/90train_negative74042.txt"  # Negative FASTA path
    train_neg_output_file = "74042_neg_protein_embeddings_fortrain.csv"  # Negative output CSV
    train_merged_output_file = "Pos_and_Neg_train_3572+74042.csv"  # Train merged output CSV

    '''Test Independent'''
    test_pos_fasta_file = "../Data/10test_positive397.txt"  # Positive test FASTA path
    test_pos_output_file = "397_pos_protein_test.csv"  # Positive test output CSV
    test_neg_fasta_file = "../Data/10test_negative8227.txt"  # Negative test FASTA path
    test_neg_output_file = "8227_neg_protein_test.csv"  # Negative test output CSV
    test_merged_output_file = "Pos_and_Neg_test_397+8227.csv"  # Test merged output CSV

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, alphabet = load_model(device)

    # Process training positive samples
    train_pos_embeddings = extract_features(train_pos_fasta_file, model, alphabet, device, batch_size=4)
    save_features_as_csv(train_pos_embeddings, train_pos_output_file)

    # Process training negative samples
    train_neg_embeddings = extract_features(train_neg_fasta_file, model, alphabet, device, batch_size=4)
    save_features_as_csv(train_neg_embeddings, train_neg_output_file)

    # Merge training positive and negative CSV files
    merge_csv_with_labels(train_pos_output_file, train_neg_output_file, train_merged_output_file)

    # Process test positive samples
    test_pos_embeddings = extract_features(test_pos_fasta_file, model, alphabet, device, batch_size=4)
    save_features_as_csv(test_pos_embeddings, test_pos_output_file)

    # Process test negative samples
    test_neg_embeddings = extract_features(test_neg_fasta_file, model, alphabet, device, batch_size=4)
    save_features_as_csv(test_neg_embeddings, test_neg_output_file)

    # Merge test positive and negative CSV files
    merge_csv_with_labels(test_pos_output_file, test_neg_output_file, test_merged_output_file)

if __name__ == "__main__":
    main()