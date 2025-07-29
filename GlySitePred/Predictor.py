import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import xgboost as xgb
from Bio import SeqIO
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from esm import pretrained
from transformers import T5Tokenizer, T5EncoderModel
import torch.hub


class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Protein Prediction Tool")

        # Initialize interface
        self.file_path = None
        self.df = None
        self.model = None
        self.sequences = None  # To store FASTA sequences and their IDs

        # Upload file button
        self.upload_button = tk.Button(root, text="Upload FASTA File", command=self.upload_file)
        self.upload_button.pack(pady=10)

        # Textbox to display file content
        self.textbox = tk.Text(root, height=15, width=50)
        self.textbox.pack(pady=10)

        # Start prediction button
        self.predict_button = tk.Button(root, text="Start Prediction", command=self.start_prediction)
        self.predict_button.pack(pady=10)

        # Textbox to display prediction results (scrollable)
        self.result_textbox = tk.Text(root, height=15, width=50)
        self.result_textbox.pack(pady=10)
        self.result_textbox.config(state=tk.DISABLED)

        # Save results button
        self.save_button = tk.Button(root, text="Save Results as CSV", command=self.save_results)
        self.save_button.pack(pady=10)

        # Load model
        self.load_model()

    def load_model(self):
        """Load XGBoost model"""
        model_path = './models/xgboost_model.json'  # Use relative path
        if os.path.exists(model_path):
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            print("Model loaded successfully")
        else:
            messagebox.showerror("Error", f"Model file '{model_path}' not found! Please ensure 'xgboost_model.json' is in the './models/' directory.")

    def upload_file(self):
        """Upload FASTA file and display its content"""
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            self.file_path = file_path

            # Display FASTA file content
            with open(file_path, 'r') as file:
                file_content = file.read()

            # Show file content in textbox
            self.textbox.delete(1.0, tk.END)
            self.textbox.insert(tk.END, file_content)

            # Extract features and store results
            self.sequences = self.parse_fasta(file_path)
            self.df = self.extract_features_from_fasta(file_path)

    def extract_features_from_fasta(self, fasta_file):
        """Extract features from FASTA file (combine two feature extraction processes)"""
        # 1. Extract first feature (using ESM-2)
        esm_embeddings = self.extract_esm_features(fasta_file)

        # 2. Extract second feature (using ProstT5)
        prostt5_embeddings = self.extract_prostt5_features(fasta_file)

        # 3. Concatenate features horizontally
        combined_embeddings = pd.concat([esm_embeddings, prostt5_embeddings], axis=1)
        return combined_embeddings

    def extract_esm_features(self, fasta_file):
        """Extract features using ESM-2"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = "./models"  # Specify download path to models directory
        os.makedirs(cache_dir, exist_ok=True)  # Ensure directory exists
        # Set torch.hub cache path
        torch.hub.set_dir(cache_dir)
        model, alphabet = pretrained.esm2_t12_35M_UR50D()
        model = model.to(device)
        model.eval()

        # Read and clean FASTA sequences
        sequences = self.parse_fasta(fasta_file)

        # Convert to model-compatible format
        batch_converter = alphabet.get_batch_converter()
        batch_output = batch_converter(sequences)
        batch_labels, batch_strs, batch_tokens = batch_output

        # Move batch data to GPU
        batch_tokens = batch_tokens.to(device)

        # Use DataLoader for batch processing
        dataset = TensorDataset(batch_tokens)
        data_loader = DataLoader(dataset, batch_size=8)

        all_embeddings = []
        for batch in data_loader:
            batch_token_sub = batch[0]

            with torch.no_grad():
                results = model(batch_token_sub, repr_layers=[12])

            # Extract protein embeddings
            embeddings = results["representations"][12]
            embeddings = embeddings.mean(dim=1)
            all_embeddings.append(embeddings)

            torch.cuda.empty_cache()

        # Combine all batch embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Convert to DataFrame and return
        embeddings_df = pd.DataFrame(all_embeddings.cpu().numpy())
        return embeddings_df

    def extract_prostt5_features(self, fasta_file):
        """Extract features using ProstT5"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "Rostlab/prostt5"  # Hugging Face model name
        cache_dir = "./models"  # Specify download path to models directory
        os.makedirs(cache_dir, exist_ok=True)  # Ensure directory exists
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, cache_dir=cache_dir)
        model = T5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        model.eval()

        # Read and clean FASTA sequences
        sequences = [self.add_space_to_sequence(seq[1]) for seq in self.parse_fasta(fasta_file)]

        # Convert to model-compatible format
        inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Use DataLoader for batch processing
        dataset = TensorDataset(input_ids, attention_mask)
        data_loader = DataLoader(dataset, batch_size=8)

        all_embeddings = []
        for batch in data_loader:
            batch_input_ids, batch_attention_mask = batch
            with torch.no_grad():
                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

            # Extract last hidden state and average token features
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings)

        # Combine all batch embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Convert to DataFrame and return
        embeddings_df = pd.DataFrame(all_embeddings.cpu().numpy())
        return embeddings_df

    def parse_fasta(self, file_path):
        """Parse FASTA file"""
        data = []
        with open(file_path, 'r') as file:
            sequences = SeqIO.parse(file, "fasta")
            for seq_record in sequences:
                data.append((seq_record.id, str(seq_record.seq)))
        return data

    def add_space_to_sequence(self, sequence):
        """Add space between each amino acid in the protein sequence"""
        return " ".join(sequence)

    def start_prediction(self):
        """Load model and perform prediction"""
        if self.df is not None:
            if self.model is None:
                messagebox.showerror("Error", "Please load the model first!")
                return

            # Perform prediction using extracted features
            X = self.df.values
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)[:, 1]

            # Display prediction results with sequence IDs
            result_text = "\n".join([f">{self.sequences[i][0]}    {'positive' if y_pred[i] == 1 else 'negative'}    {y_prob[i]:.4f} \n{self.sequences[i][1]}" for i in range(len(self.sequences))])

            # Update textbox to show prediction results
            self.result_textbox.config(state=tk.NORMAL)
            self.result_textbox.delete(1.0, tk.END)
            self.result_textbox.insert(tk.END, result_text)
            self.result_textbox.config(state=tk.DISABLED)
        else:
            messagebox.showerror("Error", "Please upload a file first!")

    def save_results(self):
        """Save prediction results as CSV file"""
        if self.model is not None and self.df is not None:
            # Predict data
            X = self.df.values
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)[:, 1]

            # Save prediction results with sequence IDs
            results = []
            for i in range(len(self.sequences)):
                results.append([self.sequences[i][0], self.sequences[i][1], 'positive' if y_pred[i] == 1 else 'negative', y_prob[i]])

            # Convert to DataFrame and save as CSV
            result_df = pd.DataFrame(results, columns=["Protein_ID", "Sequence", "Prediction", "Probability"])
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if save_path:
                result_df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Results saved to {save_path}")
        else:
            messagebox.showerror("Error", "Please perform prediction first!")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()