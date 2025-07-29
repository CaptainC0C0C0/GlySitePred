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


class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prediction Tool")

        # 初始化界面
        self.file_path = None
        self.df = None
        self.model = None
        self.sequences = None  # 用于存储FASTA序列及其ID

        # 上传文件按钮
        self.upload_button = tk.Button(root, text="上传FASTA文件", command=self.upload_file)
        self.upload_button.pack(pady=10)

        # 显示文件内容的文本框
        self.textbox = tk.Text(root, height=15, width=50)  # 增加文本框的显示高度
        self.textbox.pack(pady=10)

        # 开始预测按钮
        self.predict_button = tk.Button(root, text="开始预测", command=self.start_prediction)
        self.predict_button.pack(pady=10)

        # 显示预测结果的文本框（支持滚动查看）
        self.result_textbox = tk.Text(root, height=15, width=50)  # 可以滑动的文本框
        self.result_textbox.pack(pady=10)
        self.result_textbox.config(state=tk.DISABLED)  # 禁止直接编辑文本框内容

        # 保存结果按钮
        self.save_button = tk.Button(root, text="保存结果为CSV", command=self.save_results)
        self.save_button.pack(pady=10)

        # 加载模型
        self.load_model()

    def load_model(self):
        """加载XGBoost模型"""
        model_path = 'xgboost_model.json'  # 默认路径和文件名
        if os.path.exists(model_path):
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            print("模型加载成功")
        else:
            messagebox.showerror("错误", f"模型文件 '{model_path}' 未找到！")

    def upload_file(self):
        """上传FASTA文件并显示文件内容"""
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            self.file_path = file_path

            # 显示FASTA文件内容
            with open(file_path, 'r') as file:
                file_content = file.read()

            # 将FASTA文件内容显示在文本框中
            self.textbox.delete(1.0, tk.END)  # 清空文本框
            self.textbox.insert(tk.END, file_content)  # 显示文件内容

            # 进行特征提取并返回结果
            self.sequences = self.parse_fasta(file_path)
            self.df = self.extract_features_from_fasta(file_path)

    def extract_features_from_fasta(self, fasta_file):
        """从FASTA文件中提取特征（合并两个特征提取过程）"""
        # 1. 提取第一个特征（使用ESM-2）
        esm_embeddings = self.extract_esm_features(fasta_file)

        # 2. 提取第二个特征（使用ProstT5）
        prostt5_embeddings = self.extract_prostt5_features(fasta_file)

        # 3. 横向合并两个特征
        combined_embeddings = pd.concat([esm_embeddings, prostt5_embeddings], axis=1)  # 横向合并

        return combined_embeddings

    def extract_esm_features(self, fasta_file):
        """使用ESM-2提取特征"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, alphabet = pretrained.esm2_t12_35M_UR50D()
        model = model.to(device)
        model.eval()

        # 读取FASTA文件并清理序列
        sequences = self.parse_fasta(fasta_file)

        # 转换为模型可以处理的格式
        batch_converter = alphabet.get_batch_converter()
        batch_output = batch_converter(sequences)
        batch_labels, batch_strs, batch_tokens = batch_output

        # 将批处理数据移动到GPU
        batch_tokens = batch_tokens.to(device)

        # 使用 DataLoader 来分批处理数据
        dataset = TensorDataset(batch_tokens)
        data_loader = DataLoader(dataset, batch_size=8)

        all_embeddings = []
        for batch in data_loader:
            batch_token_sub = batch[0]

            with torch.no_grad():
                results = model(batch_token_sub, repr_layers=[12])

            # 提取蛋白质嵌入
            embeddings = results["representations"][12]
            embeddings = embeddings.mean(dim=1)  # 对每个序列，所有层的嵌入取平均值
            all_embeddings.append(embeddings)

            torch.cuda.empty_cache()  # 清空缓存以释放内存

        # 合并所有批次的嵌入
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # 转换为DataFrame并返回
        embeddings_df = pd.DataFrame(all_embeddings.cpu().numpy())
        return embeddings_df

    def extract_prostt5_features(self, fasta_file):
        """使用ProstT5提取特征"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "D:/bioModel/ProstT5"  # 模型路径
        tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(model_path).to(device)
        model.eval()

        # 读取FASTA文件并清理序列
        sequences = [self.add_space_to_sequence(seq[1]) for seq in self.parse_fasta(fasta_file)]

        # 转换为模型可处理的格式
        inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 使用 DataLoader 分批处理数据
        dataset = TensorDataset(input_ids, attention_mask)
        data_loader = DataLoader(dataset, batch_size=8)

        all_embeddings = []
        for batch in data_loader:
            batch_input_ids, batch_attention_mask = batch
            with torch.no_grad():
                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

            # 提取最后一层隐藏状态，并对 token 特征取平均值
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings)

        # 合并所有批次的嵌入
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # 转换为DataFrame并返回
        embeddings_df = pd.DataFrame(all_embeddings.cpu().numpy())
        return embeddings_df

    def parse_fasta(self, file_path):
        """解析FASTA文件"""
        data = []
        with open(file_path, 'r') as file:
            sequences = SeqIO.parse(file, "fasta")
            for seq_record in sequences:
                data.append((seq_record.id, str(seq_record.seq)))
        return data

    def add_space_to_sequence(self, sequence):
        """
        在蛋白质序列的每个氨基酸之间插入空格。
        :param sequence: 原始的蛋白质序列字符串
        :return: 每个氨基酸之间带空格的新字符串
        """
        return " ".join(sequence)

    def start_prediction(self):
        """加载模型并进行预测"""
        if self.df is not None:
            if self.model is None:
                messagebox.showerror("错误", "请先加载模型！")
                return

            # 使用提取的特征进行预测
            X = self.df.values  # 获取特征数据
            y_pred = self.model.predict(X)

            # 获取预测的概率
            y_prob = self.model.predict_proba(X)[:, 1]  # 假设我们关心的类别是1（positive）

            # 将预测结果与序列ID一起显示
            result_text = "\n".join([f">{self.sequences[i][0]}    {'positive' if y_pred[i] == 1 else 'negative'}    {y_prob[i]:.4f} \n{self.sequences[i][1]}" for i in range(len(self.sequences))])

            # 更新文本框显示预测结果
            self.result_textbox.config(state=tk.NORMAL)  # 使文本框可编辑
            self.result_textbox.delete(1.0, tk.END)  # 清空文本框
            self.result_textbox.insert(tk.END, result_text)  # 插入预测结果
            self.result_textbox.config(state=tk.DISABLED)  # 禁止编辑文本框内容
        else:
            messagebox.showerror("错误", "请先上传文件！")

    def save_results(self):
        """保存预测结果为CSV文件"""
        if self.model is not None and self.df is not None:
            # 预测数据
            X = self.df.values
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)[:, 1]

            # 将预测结果与序列ID一起保存
            results = []
            for i in range(len(self.sequences)):
                results.append([self.sequences[i][0], self.sequences[i][1], 'positive' if y_pred[i] == 1 else 'negative', y_prob[i]])

            # 转换为DataFrame并保存为CSV
            result_df = pd.DataFrame(results, columns=["Protein_ID", "Sequence", "Prediction", "Probability"])
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if save_path:
                result_df.to_csv(save_path, index=False)
                messagebox.showinfo("保存成功", f"结果已保存到 {save_path}")
        else:
            messagebox.showerror("错误", "请先进行预测！")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
