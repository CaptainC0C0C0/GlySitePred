import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5Tokenizer, T5EncoderModel

# Set a local cache directory for model downloads
cache_dir = os.path.join(os.path.dirname(__file__), "torch_cache")
os.environ["TORCH_HOME"] = cache_dir

HF_MODEL_ID = "Rostlab/ProstT5"
LOCAL_MODEL_PATH = "../BioModel/ProstT5"


def ensure_prostt5_available(model_path: str, hf_model_id: str):
    """
    Ensure ProstT5 model and tokenizer are available locally. If not, download from HF and save.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading from Hugging Face ({hf_model_id})...")
        os.makedirs(model_path, exist_ok=True)
        tokenizer = T5Tokenizer.from_pretrained(hf_model_id, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(hf_model_id)
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        print(f"Model downloaded and saved to {model_path}")
    else:
        print(f"Model path found: {model_path}")


def load_model_and_tokenizer(device: torch.device, model_path: str):
    """
    Load the ProstT5 tokenizer and T5EncoderModel from local path and move model to device.
    Returns (model, tokenizer).
    """
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def add_space_to_sequence(sequence: str) -> str:
    """
    Insert a space between each amino acid character, e.g. "MKV" -> "M K V".
    ProstT5 expects tokenized amino acids separated by spaces.
    """
    # remove whitespace/newlines then join with spaces
    seq = sequence.replace("\n", "").replace("\r", "").strip()
    return " ".join(list(seq))


def parse_fasta(file_path: str):
    """
    Parse a FASTA-like file and return list of (name, spaced_sequence).
    Works with single-line or multi-line sequences under headers starting with '>'.
    """
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    with open(file_path, "r") as f:
        protein_name = None
        seq_lines = []
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if protein_name is not None:
                    seq = "".join(seq_lines)
                    data.append((protein_name, add_space_to_sequence(seq)))
                protein_name = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        # last record
        if protein_name is not None:
            seq = "".join(seq_lines)
            data.append((protein_name, add_space_to_sequence(seq)))
    return data


def extract_features_prostt5(fasta_file: str, model: T5EncoderModel, tokenizer: T5Tokenizer,
                             device: torch.device, batch_size: int = 8, max_length: int = 512):
    """
    Tokenize sequences with ProstT5 tokenizer and extract pooled embeddings by averaging token embeddings
    (taking attention_mask into account). Returns a tensor of shape (N, hidden_size).
    """
    parsed = parse_fasta(fasta_file)
    if len(parsed) == 0:
        print(f"No sequences parsed from {fasta_file}")
        return torch.empty((0, model.config.hidden_size))

    # sequences are already spaced
    names = [p[0] for p in parsed]
    sequences = [p[1] for p in parsed]

    # Tokenize all sequences (padding/truncation)
    inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_embeddings = []
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for batch in loader:
            batch_input_ids, batch_attention_mask = batch
            batch_input_ids = batch_input_ids.to(model_device)
            batch_attention_mask = batch_attention_mask.to(model_device)

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            # last_hidden_state shape: (batch, seq_len, hidden_size)
            last_hidden = outputs.last_hidden_state

            # Compute mask-aware mean pooling over token dimension
            mask = batch_attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            masked_sum = (last_hidden * mask).sum(dim=1)  # sum over seq_len -> (batch, hidden_size)
            denom = mask.sum(dim=1).clamp(min=1)  # avoid div by zero
            pooled = masked_sum / denom  # (batch, hidden_size)

            all_embeddings.append(pooled)
            # free some GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


def save_features_as_csv(embeddings: torch.Tensor, output_file: str):
    """
    Save embeddings tensor to CSV. Creates output directories if needed.
    """
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"Created output directory: {out_dir}")

    embeddings_np = embeddings.cpu().numpy()
    df = pd.DataFrame(embeddings_np)
    df.to_csv(output_file, index=False)
    print(f"Saved features to {output_file} (shape {embeddings_np.shape})")


def merge_csv_with_labels(pos_file: str, neg_file: str, output_file: str):
    """
    Read pos and neg CSVs, add 'Label' (1 for pos, 0 for neg), concatenate and save.
    """
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"Created directory for merged output: {out_dir}")

    pos_df = pd.read_csv(pos_file)
    neg_df = pd.read_csv(neg_file)
    pos_df["Label"] = 1
    neg_df["Label"] = 0
    merged = pd.concat([pos_df, neg_df], ignore_index=True)
    merged.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file} (pos: {len(pos_df)} + neg: {len(neg_df)})")


def main():
    # --- file mapping requested by user (ProstT5 filenames) ---
    # Train
    train_pos_fasta_file = "../Data/90train_positive3572.txt"
    train_pos_output_file = "ProstT5_3572_pos_protein_embeddings_forval.csv"
    train_neg_fasta_file = "../Data/90train_negative74042.txt"
    train_neg_output_file = "ProstT5_74042_neg_protein_embeddings_fortrain.csv"
    train_merged_output_file = "ProstT5_Pos_and_Neg_train_3572+74042.csv"

    # Test independent
    test_pos_fasta_file = "../Data/10test_positive397.txt"
    test_pos_output_file = "ProstT5_397_pos_protein_test.csv"
    test_neg_fasta_file = "../Data/10test_negative8227.txt"
    test_neg_output_file = "ProstT5_8227_neg_protein_test.csv"
    test_merged_output_file = "ProstT5_Pos_and_Neg_test_397+8227.csv"

    # Ensure model presence and load
    ensure_prostt5_available(LOCAL_MODEL_PATH, HF_MODEL_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(device, LOCAL_MODEL_PATH)

    # Process training positive
    print("Extracting train positive embeddings...")
    train_pos_embeddings = extract_features_prostt5(train_pos_fasta_file, model, tokenizer, device, batch_size=4)
    save_features_as_csv(train_pos_embeddings, train_pos_output_file)

    # Process training negative
    print("Extracting train negative embeddings...")
    train_neg_embeddings = extract_features_prostt5(train_neg_fasta_file, model, tokenizer, device, batch_size=4)
    save_features_as_csv(train_neg_embeddings, train_neg_output_file)

    # Merge train
    merge_csv_with_labels(train_pos_output_file, train_neg_output_file, train_merged_output_file)

    # Process test positive
    print("Extracting test positive embeddings...")
    test_pos_embeddings = extract_features_prostt5(test_pos_fasta_file, model, tokenizer, device, batch_size=4)
    save_features_as_csv(test_pos_embeddings, test_pos_output_file)

    # Process test negative
    print("Extracting test negative embeddings...")
    test_neg_embeddings = extract_features_prostt5(test_neg_fasta_file, model, tokenizer, device, batch_size=4)
    save_features_as_csv(test_neg_embeddings, test_neg_output_file)

    # Merge test
    merge_csv_with_labels(test_pos_output_file, test_neg_output_file, test_merged_output_file)

    print("All done.")


if __name__ == "__main__":
    main()
