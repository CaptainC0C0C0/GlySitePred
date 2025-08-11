import numpy as np
import pandas as pd

def one_hot_encode_sequences(aa_sequences, amino_acids='ACDEFGHIKLMNPQRSTVWYO'):
    # Create a dictionary that maps each amino acid to its one-hot vector
    amino_acid_to_one_hot = {amino_acid: np.eye(len(amino_acids))[i]
                             for i, amino_acid in enumerate(amino_acids)}

    # Initialize list to store one-hot encoded sequences
    one_hot_encoded_sequences = []

    # Encode each sequence
    for sequence in aa_sequences:
        one_hot_encoded_sequence = np.zeros((len(sequence), len(amino_acids)))
        for i, amino_acid in enumerate(sequence):
            if amino_acid in amino_acid_to_one_hot:
                one_hot_encoded_sequence[i] = amino_acid_to_one_hot[amino_acid]
            else:
                raise ValueError(f"Unknown amino acid: '{amino_acid}'")
        one_hot_encoded_sequences.append(one_hot_encoded_sequence)

    return np.array(one_hot_encoded_sequences).reshape(len(one_hot_encoded_sequences), -1)

# Read FASTA file with labels
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
file_path = '../Data/90train_positive3572.txt'

# Read sequences and labels
sequences, labels = read_fasta_with_labels(file_path)

# Extract features
features = one_hot_encode_sequences(sequences, amino_acids='ACDEFGHIKLMNPQRSTVWYO')  # includes 'O' as a special amino acid

# Convert to DataFrame
df = pd.DataFrame(features)
df['Label'] = labels

# Save to file
output_file = '3572_onehot_pos.csv'
df.to_csv(output_file, index=False)

print(f"Feature extraction complete. Results saved to {output_file}")