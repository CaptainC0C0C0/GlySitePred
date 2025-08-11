import numpy as np
import pandas as pd
from collections import Counter


def aac_feature(sequences, amino_acids='ACDEFGHIKLMNPQRSTVWYO'):
    """Extract amino-acid composition (AAC) features."""
    # Create an amino-acid set for quick lookup
    valid_aa = set(amino_acids)

    # Initialize feature list
    features = []

    for seq in sequences:
        # Check for invalid amino acids
        for aa in seq:
            if aa not in valid_aa:
                raise ValueError(f"Unknown amino acid: '{aa}'")

        # Compute amino-acid composition
        counter = Counter(seq)
        total = len(seq)
        aac_vector = [counter.get(aa, 0) / total for aa in amino_acids]
        features.append(aac_vector)

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
file_path = '../Data/10test_positive397.txt'

# Read sequences and labels
sequences, labels = read_fasta_with_labels(file_path)

# Extract AAC features
amino_acids = 'ACDEFGHIKLMNPQRSTVWYO'  # 21 amino acids, including special 'O'
features = aac_feature(sequences, amino_acids=amino_acids)

# Convert to DataFrame
df = pd.DataFrame(features, columns=[f"AAC_{aa}" for aa in amino_acids])
df['Label'] = labels

# Save to file
output_file = '397_aac_pos_features.csv'
df.to_csv(output_file, index=False)

print(f"Feature extraction complete. Results saved to {output_file}")