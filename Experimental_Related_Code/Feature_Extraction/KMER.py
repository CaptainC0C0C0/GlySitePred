import numpy as np
import pandas as pd
from itertools import product
from collections import Counter


def kmer_feature(sequences, k=3, amino_acids='ACDEFGHIKLMNPQRSTVWYO'):
    """Extract k-mer frequency features."""
    # Generate all possible k-mers
    kmers = [''.join(p) for p in product(amino_acids, repeat=k)]
    kmer_index = {kmer: idx for idx, kmer in enumerate(kmers)}

    # Validate amino acids
    valid_aa = set(amino_acids)
    for seq in sequences:
        for aa in seq:
            if aa not in valid_aa:
                raise ValueError(f"Unknown amino acid: '{aa}'")

    # Compute k-mer frequencies
    features = []
    for seq in sequences:
        seq_len = len(seq)
        if seq_len < k:
            raise ValueError(f"Sequence length {seq_len} is less than k={k}")

        # Extract k-mers and count occurrences
        seq_kmers = [seq[i:i + k] for i in range(seq_len - k + 1)]
        counter = Counter(seq_kmers)
        total = len(seq_kmers)

        # Build feature vector
        feature = np.zeros(len(kmers))
        for kmer, count in counter.items():
            feature[kmer_index[kmer]] = count / total
        features.append(feature)

    return np.array(features)


# Read FASTA file with labels (same as original)
def read_fasta_with_labels(file_path):
    sequences = []
    labels = []
    with open(file_path, 'r') as file:
        label = None
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    labels.append(label)
                label = 'Positive' if 'Positive' in line else 'Negative'
                sequence = ''
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
            labels.append(label)
    return sequences, labels


# File path
file_path = '../Data/10test_negative8227.txt'

# Read data
sequences, labels = read_fasta_with_labels(file_path)

# Extract k-mer features (example uses k=3)
amino_acids = 'ACDEFGHIKLMNPQRSTVWYO'  # 21 amino acids including 'O'
k_value = 3  # Set to 2, 3, 4, etc. as needed
features = kmer_feature(sequences, k=k_value, amino_acids=amino_acids)

# Generate column names (e.g., AA, AC, AD, ...)
kmer_columns = [''.join(p) for p in product(amino_acids, repeat=k_value)]

# Create DataFrame
df = pd.DataFrame(features, columns=[f"Kmer_{col}" for col in kmer_columns])
df['Label'] = labels

# Save results
output_file = f'8227_kmer{k_value}_neg_features.csv'
df.to_csv(output_file, index=False)

print(f"Feature extraction complete. Generated {len(kmer_columns)}-dimensional k-mer features. Results saved to {output_file}")