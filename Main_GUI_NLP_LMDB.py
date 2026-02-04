from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter as tk


# ===============================
# Core Python Libraries
# ===============================
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from scipy.special import expit  # sigmoid

# ===============================
# Data Visualization
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image, ImageTk
import cv2
# ===============================
# System & Utilities
# ===============================
import os
import joblib
import numpy as np
from tqdm import tqdm
import re

# ===============================
# NLP: NLTK 
# ===============================
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.util import ngrams

# NLTK Downloads
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ===============================
# Scikit-learn: Preprocessing, Models, Metrics
# ===============================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib, os
from sklearn.ensemble import RandomForestClassifier
from roft import RandomObliqueForestTrees
# ===============================
# TensorFlow / Keras
# ===============================
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ===============================
# Transformers (Hugging Face)
# ===============================
import torch
# ===============================
# Custom Modules
# ===============================
from metrics_calculator import MetricsCalculator
from transformers import CanineTokenizer, CanineModel
from graphs import GraphPlotter
from imblearn.over_sampling import SMOTE

# ===============================
# Model Directory
# ===============================
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)



def upload_dataset(unilable):
    try:
        """Load the dataset from a CSV file"""
        global file_path,df
        text.delete('1.0', END)
        file_path = filedialog.askopenfilename(initialdir = "Dataset",
                                            #    filetypes=[("CSV files", "*.csv")]
                                               )

        df = pd.read_csv(file_path)
        text.insert(END,str(df.head())+"\n\n")
    except Exception as e:
        # unilable.config(text=)
        text.tag_configure("error", foreground="red")
        text.insert(END, "Invalid file type, please upload CSV only\n", "error")



def preprocess_data(df, save_path=None, target_cols=None):

    global label_encoders
    label_encoders = {}  # dictionary to hold encoders for each target column

    if save_path and os.path.exists(save_path):
        print(f"Loading existing preprocessed file: {save_path}")
        df = pd.read_csv(save_path)
    else:
        print("Preprocessing data" + (f" and saving to: {save_path}" if save_path else " (no saving)"))
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            text = str(text).lower()
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
            return ' '.join(tokens)

        # Separate target columns
        target_df = None
        if target_cols:
            existing_targets = [col for col in target_cols if col in df.columns]
            target_df = df[existing_targets].copy()
            df = df.drop(columns=existing_targets)

        # Process text columns
        text_columns = df.select_dtypes(include='object').columns
        for col in text_columns:
            df[f'processed_{col}'] = df[col].apply(clean_text)

        # Drop original text columns
        df.drop(columns=text_columns, inplace=True)

        # Reattach target columns
        if target_df is not None:
            for col in target_df.columns:
                df[col] = target_df[col]

        # Save only if path is specified
        if save_path:
            df.to_csv(save_path, index=False)

    # Select processed and numerical columns
    processed_text_cols = [col for col in df.columns if col.startswith('processed_')]
    non_text_cols = [col for col in df.columns if col not in processed_text_cols + (target_cols if target_cols else [])]

    # Join processed text columns into one string
    X_text = df[processed_text_cols].astype(str).agg(' '.join, axis=1)

    # Combine with numerical columns if any
    X_numeric = df[non_text_cols].values if non_text_cols else None
    if X_numeric is not None and len(X_numeric) > 0:
        X = [f"{text} {' '.join(map(str, numeric))}" for text, numeric in zip(X_text, X_numeric)]
    else:
        X = X_text.tolist()

    # Encode multiple target columns
    Y_dict = {}
    if target_cols:
        for col in target_cols:
            if col in df.columns:
                le = LabelEncoder()
                Y_dict[col] = le.fit_transform(df[col])
                label_encoders[col] = le

    return X, Y_dict

def plot_target_distributions(Y_dict):

    # Convert to DataFrame
    y_df = pd.DataFrame(Y_dict)

    # Loop through each target column
    for col in y_df.columns:

        # Print total number of rows
        total_rows = len(y_df[col])
        print(f"{col} → Total Rows: {total_rows}")

        # Plot count distribution
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x=y_df[col])

        # Add count labels on each bar
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f'{height}',                     # text
                (p.get_x() + p.get_width()/2, height),  # position
                ha='center', va='bottom', 
                fontsize=10, fontweight='bold'
            )

        plt.title(f'Class Distribution: {col}')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()


def Preprocess_Dataset_button():
    global df, X, Y_dict
    global metrics_calculator_dict,target_cols

    text.delete('1.0', END)
    MODEL_DIR = "model"

    target_cols = ["churn_risk_score", "complaint_status"]
    X, Y_dict = preprocess_data(
        df,
        save_path="model/cleaned_data.csv",
        target_cols=target_cols
    )

    metrics_calculator_dict = {}

    text.insert(END, "Extracted Label Classes:\n\n")

    for col, le in label_encoders.items():

        labels = list(le.classes_)    # class names list

        # Print readable summary
        text.insert(END, f"Target: {col}\n")
        for idx, class_name in enumerate(labels):
            text.insert(END, f"  {idx}: {class_name}\n")
        text.insert(END, "\n")


        label_file = os.path.join(MODEL_DIR, f"labels_{col}.npy")
        np.save(label_file, np.array(labels), allow_pickle=True)


        metrics_calculator_dict[col] = MetricsCalculator(labels, text_widget=text)

    text.insert(END, "Labels saved successfully!\n")

    # Plot distributions
    plot_target_distributions(Y_dict)


def eda_nlp_analysis():
    global df,X, Y_dict
    text.delete('1.0', END)

    X_text=X
    num_words=100
    top_n_words=20
    
    text.insert(END,"Generating NLP EDA Visualizations..."+"\n\n")

    # Flatten all tokens from all texts
    all_tokens = [word for doc in X_text for word in word_tokenize(doc)]

    # --- 1. WordCloud ---
    word_freq = Counter(all_tokens)
    wc = WordCloud(width=800, height=400, max_words=num_words, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Top {num_words} Words - WordCloud")
    plt.show()

    # --- 2. Top-N Frequent Words ---
    common_words = word_freq.most_common(top_n_words)
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title(f"Top {top_n_words} Most Frequent Words")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.show()

    # --- 3. Document Length Histogram ---
    doc_lengths = [len(word_tokenize(doc)) for doc in X_text]
    plt.figure(figsize=(10, 5))
    sns.histplot(doc_lengths, bins=20, kde=True, color='teal')
    plt.title("Distribution of Document Lengths (in words)")
    plt.xlabel("Number of Words per Document")
    plt.ylabel("Frequency")
    plt.show()

    # --- 4. POS Tag Frequency ---
    all_pos = [tag for _, tag in pos_tag(all_tokens)]
    pos_counts = Counter(all_pos).most_common()
    pos_tags, pos_freqs = zip(*pos_counts)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(pos_tags), y=list(pos_freqs), palette="coolwarm")
    plt.title("Part of Speech (POS) Tag Frequency")
    plt.xlabel("POS Tag")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()

    # --- 5. Bigram Frequency Plot ---
    bigrams = list(ngrams(all_tokens, 2))
    bigram_freq = Counter(bigrams).most_common(top_n_words)
    bigram_labels = [' '.join(b) for b, _ in bigram_freq]
    bigram_counts = [count for _, count in bigram_freq]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=bigram_counts, y=bigram_labels, palette="magma")
    plt.title(f"Top {top_n_words} Bigrams")
    plt.xlabel("Count")
    plt.ylabel("Bigram")
    plt.show()




def canine_feature_extraction(
        texts,
        model_name='google/canine-s',
        batch_size=32,
        pooling='cls',  # or 'mean'
        device=None):

    tokenizer = CanineTokenizer.from_pretrained(model_name)
    model = CanineModel.from_pretrained(model_name)

    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting CANINE embeddings"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            output = model(**enc)

        # output.last_hidden_state: sequence of hidden states for each character’s embedding after upsampling
        # output.pooler_output: the pooled [CLS]-style output (for classification / sentence-level features)
        if pooling == 'cls':
            embs = output.pooler_output
        elif pooling == 'mean':
            # mean over sequence dimension
            embs = output.last_hidden_state.mean(dim=1)
        else:
            raise ValueError("pooling must be 'cls' or 'mean'")

        all_embs.append(embs.cpu().numpy())

    X = np.vstack(all_embs)
    return X, model

def feature_extraction(X_text, method='CANNIE_Embeddings', model_dir='model', is_train=True):
    x_file = os.path.join(model_dir, f'X_{method}.pkl')

    text.insert(END, f"[INFO] Feature extraction method: {method}, Train mode: {is_train}"+"\n\n")

    if is_train:
        if os.path.exists(x_file):
            text.insert(END, f"[INFO] Loading cached embeddings from {x_file}"+"\n\n")
            X = joblib.load(x_file)
        else:
            text.insert(END, "[INFO] Computing embeddings..."+"\n\n")
            X, model = canine_feature_extraction(X_text, pooling='mean')
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(X, x_file)
    else:
        text.insert(END, "[INFO] Performing embedding extraction for testing..."+"\n\n")
        X, model = canine_feature_extraction(X_text, pooling='mean')

    return X
    
def feature_extraction_button():
    global X, features
    global features_dict, labels_dict, target_cols

    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    text.delete('1.0', END)

    features = feature_extraction(X, method='CANNIE_Embeddings', is_train=True)

    original_features = {}
    original_labels = {}

    # ---------------- ORIGINAL DATA ----------------
    for i, (key, y) in enumerate(Y_dict.items(), start=1):
        text.insert(END, f"\n🔹 Processing '{key}' (Original)...\n\n")

        original_features[f'features{i}'] = features
        original_labels[f'Y{i}'] = y

        text.insert(END, f"Dataset: features{i}.shape = {features.shape}, Y{i}.shape = {y.shape}\n\n")

    resampled_features = {}
    resampled_labels = {}

    TARGET_SIZE = 500  

    for idx, name in enumerate(target_cols, start=1):

        X_target = original_features[f'features{idx}']
        y_target = original_labels[f'Y{idx}']

        unique_classes, counts = np.unique(y_target, return_counts=True)
        class_counts = dict(zip(unique_classes, counts))

        text.insert(END, f"Original Class Counts for {name}: {class_counts}\n")

        undersample_strategy = {cls: min(cnt, TARGET_SIZE) for cls, cnt in class_counts.items()}

        rus = RandomUnderSampler(sampling_strategy=undersample_strategy)
        X_final, y_final = rus.fit_resample(X_target, y_target)

        text.insert(END, f"After Undersampling: {dict(zip(*np.unique(y_final, return_counts=True)))}\n")

        resampled_features[name] = X_final
        resampled_labels[name] = y_final

        text.insert(END, f"Final Shape for {name}: {X_final.shape}, {y_final.shape}\n\n")

    features_dict = resampled_features
    labels_dict = resampled_labels


def train_ml_models(Algorithm_prefix, features_dict, Y_dict, algorithm):
    ml_models = {}


    model_mapping = {
        "Ridge": RidgeClassifier,
        "NC": NearestCentroid,
        "RBM": BernoulliRBM,
        "Proposed": RandomObliqueForestTrees,   
    }

    if algorithm not in model_mapping:
        raise ValueError(f"Unknown algorithm: {algorithm}")


    for target_name, y_encoded in Y_dict.items():

        X = features_dict[target_name]
        model_cls = model_mapping[algorithm]


        if algorithm == "Ridge":
            mdl = model_cls(alpha=1.0)

        elif algorithm == "NC":
            mdl = model_cls()

        elif algorithm == "Proposed":
            mdl = model_cls()     

        elif algorithm == "RBM":
            rbm = BernoulliRBM(
                n_components=32,
                learning_rate=0.01,
                n_iter=2,
                batch_size=16
            )
            classifier = RidgeClassifier()
            mdl = Pipeline([
                ("rbm", rbm),
                ("classifier", classifier)
            ])

  
        model_path = f"model/CANNIE_Embeddings_{target_name}_{algorithm}_model.pkl"
        algo_name = f"{Algorithm_prefix} {algorithm} [{target_name}]"


        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2,
            random_state=42, stratify=y_encoded
        )


        print(f"[TRAIN] {algorithm} model → Target: {target_name}")
        mdl.fit(X, y_encoded)
        joblib.dump(mdl, model_path, compress=("lzma", 3))
        y_pred = mdl.predict(X_test)

        try:
            y_score = mdl.predict_proba(X_test)
        except:
            y_score = None


        if target_name in metrics_calculator_dict:
            metrics_calculator_dict[target_name].calculate_metrics(
                algo_name, y_pred, y_test, y_score
            )

        # store model in dict
        ml_models[f"{target_name}_{algorithm}"] = mdl

    return ml_models
    
def existing_classifier1():
    text.delete('1.0', END)
   
    global features_dict,labels_dict
    models = train_ml_models("CANNIE_Embeddings", features_dict, labels_dict, "Ridge")

def existing_classifier2():
    text.delete('1.0', END)

    global features_dict,labels_dict
    models = train_ml_models("CANNIE_Embeddings", features_dict, labels_dict, "NC")

def existing_classifier3():
    text.delete('1.0', END)

    global features_dict,labels_dict
    models = train_ml_models("CANNIE_Embeddings", features_dict, labels_dict, "RBM")



def Proposed_classifier():
    text.delete('1.0', END)

    global features_dict,labels_dict
    models = train_ml_models("CANNIE_Embeddings", features_dict, labels_dict, "Proposed")


def Prediction():
    text.delete('1.0', END)

    import re
    from pathlib import Path


    filename = filedialog.askopenfilename(initialdir="Dataset")
    df_test1 = pd.read_csv(filename)
    df_result = df_test1.copy()

    df_test, _ = preprocess_data(df_test1)
    features_test = feature_extraction(df_test, method='CANNIE_Embeddings', is_train=None)


    model_dir = Path("model")
    model_files = list(model_dir.glob("CANNIE_Embeddings_*_Proposed_model.pkl"))


    for model_file in model_files:
        # Extract <Target>
        match = re.search(r"CANNIE_Embeddings_(.*?)_Proposed_model\.pkl", model_file.name)
        if not match:
            continue

        target_name = match.group(1).strip()
        text.insert(END, f"\n Loading Proposed model for: {target_name}\n")


        label_file = model_dir / f"labels_{target_name}.npy"
        labels = np.load(label_file, allow_pickle=True)
        model = joblib.load(model_file)

        y_pred = model.predict(features_test)
        mapped = [labels[i] for i in y_pred]

        pred_col = f"Predicted_{target_name}"
        df_result[pred_col] = mapped


    pred_cols = [c for c in df_result.columns if c.startswith("Predicted_")]

    for i, row in df_result.iterrows():
        text.insert(END, f"\n\nRow {i+1}:\n")

        for col in df_result.columns:
            if col not in pred_cols:
                text.insert(END, f"{col}: {row[col]}\n")

        for col in pred_cols:
            text.insert(END, f"{col}: {row[col]}\n")

        text.insert(END, "\n")

def graph():
    text.delete('1.0', END)

    for target_name, calculator in metrics_calculator_dict.items():

        if calculator.metrics_df.empty:
            text.insert(END, f"No metrics found for {target_name}\n\n")
            continue

        # Copy main table
        df_h = calculator.metrics_df[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
        df_h = df_h.round(3)


        first_alg = df_h['Algorithm'].iloc[0]
        detected_name = re.search(r'\[(.*?)\]', first_alg)
        if detected_name:
            extracted_target = detected_name.group(1)
        else:
            extracted_target = target_name


        text.insert(END, f"=== Performance for {extracted_target} ===\n\n")
        df_h['Algorithm'] = df_h['Algorithm'].str.replace(r'\s*\[.*?\]', '', regex=True)
        text.insert(END, df_h.to_string(index=False))
        text.insert(END, "\n\n")

        
import hashlib
import lmdb
import json
import hashlib
import tkinter as tk
from tkinter import messagebox

# -----------------------------
# LMDB CONNECTION
# -----------------------------
def connect_lmdb():
    return lmdb.open(
        "user_auth_db",
        map_size=10 * 1024 * 1024,  # 10 MB
        max_dbs=1,
        lock=True
    )

# -----------------------------
# HASH PASSWORD
# -----------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# -----------------------------
# SIGNUP FUNCTIONALITY (LMDB)
# -----------------------------
def signup(role):

    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                env = connect_lmdb()
                hashed_pw = hash_password(password)
                user_key = f"user:{username}".encode()

                with env.begin(write=True) as txn:

                    # Check if user already exists
                    existing = txn.get(user_key)
                    if existing:
                        messagebox.showerror("Error", "User already exists!")
                        return

                    # Prepare user data
                    user_data = {
                        "username": username,
                        "password": hashed_pw,
                        "role": role
                    }

                    # Store JSON object as value
                    txn.put(user_key, json.dumps(user_data).encode())

                messagebox.showinfo("Success", f"{role} Signup Successful!")
                signup_window.destroy()

            except Exception as e:
                messagebox.showerror("Error", f"LMDB Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    # Tkinter UI
    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x400")
    signup_window.title(f"{role} Signup")

    tk.Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)

    tk.Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# -----------------------------
# LOGIN FUNCTIONALITY (LMDB)
# -----------------------------
def login(role):

    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                env = connect_lmdb()
                hashed_pw = hash_password(password)
                user_key = f"user:{username}".encode()

                with env.begin() as txn:
                    stored = txn.get(user_key)

                    if not stored:
                        messagebox.showerror("Error", "User not found!")
                        return

                    # Load stored JSON
                    user_data = json.loads(stored.decode())

                    if user_data["password"] == hashed_pw and user_data["role"] == role:
                        messagebox.showinfo("Success", f"{role} Login Successful!")
                        login_window.destroy()

                        if role == "Admin":
                            show_admin_buttons()
                        else:
                            show_user_buttons()
                    else:
                        messagebox.showerror("Error", "Invalid Credentials!")

            except Exception as e:
                messagebox.showerror("Error", f"LMDB Error: {e}")

        else:
            messagebox.showerror("Error", "Please enter all fields!")

    # Tkinter UI
    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    tk.Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)
# Clear buttons function
def clear_buttons():
    for widget in main.place_slaves():
        if isinstance(widget, tkinter.Button):
            widget.destroy()

# Admin Button Functions
def show_admin_buttons():
    clear_buttons()
    ff = ('times', 12, 'bold')
    uploadButton = Button(main, text="Dataset", command=lambda:upload_dataset(title),bg='light pink')
    uploadButton.place(x=1100,y=200)
    uploadButton.config(font=ff)

    processButton = Button(main, text="Preprocessing", command=Preprocess_Dataset_button,bg='light pink')
    processButton.place(x=1100,y=250)
    processButton.config(font=ff)

    Button1 = Button(main, text="EDA", command=eda_nlp_analysis,bg='light pink')
    Button1.place(x=1100,y=300)
    Button1.config(font=ff) 

    Button1 = Button(main, text="CANNIE Features", command=feature_extraction_button,bg='light pink')
    Button1.place(x=1100,y=350)
    Button1.config(font=ff) 

    Button1 = Button(main, text="Ridge Classifier", command=existing_classifier1,bg='light pink')
    Button1.place(x=1100,y=400)
    Button1.config(font=ff)

    Button1 = Button(main, text="NC Classifier", command=existing_classifier2,bg='light pink')
    Button1.place(x=1100,y=450)
    Button1.config(font=ff)

    Button1 = Button(main, text="RBM Classifier", command=existing_classifier3,bg='light pink')
    Button1.place(x=1100,y=500)
    Button1.config(font=ff)

    Button1 = Button(main, text="Proposed Classifier", command=Proposed_classifier,bg='light pink')
    Button1.place(x=1100,y=550)
    Button1.config(font=ff)
    
    Button1 = Button(main, text="Comparision Tables", command=graph,bg='light pink')
    Button1.place(x=1100,y=600)
    Button1.config(font=ff)
    
    tk.Button(main, text="Logout", command=show_login_screen, font=font1, bg="red").place(x=1100, y=650)

def show_user_buttons():
    font1 = ('times', 13, 'bold')
    clear_buttons()
    Button1 = Button(main, text="Prediction", command=Prediction,bg='light pink')
    Button1.place(x=1100,y=500)
    Button1.config(font=ff)


    tk.Button(main, text="Logout", command=show_login_screen, font=font1, bg="red").place(x=1100, y=600)

def show_login_screen():
    clear_buttons()
    font1 = ('times', 14, 'bold')

    tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=1, bg='steelblue').place(x=100, y=100)
    tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=1, bg='steelblue').place(x=400, y=100)
    tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='steelblue').place(x=700, y=100)
    tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=1, bg='steelblue').place(x=1000, y=100)

def close():
    main.destroy()



main = tkinter.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

# ------------------ Load and place background ------------------
bg_image = Image.open("background.png")  
bg_image = bg_image.resize((screen_width, screen_height))
bg_photo = ImageTk.PhotoImage(bg_image)

background_label = tk.Label(main, image=bg_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


font = ('times', 18, 'bold')
global title
title = Label(main, text='Classification of Customer Churn Using Character Architecture Driven Neural Encoders')
title.config(bg='steelblue', fg='black')
title.config(font=font)
title.config(height=3, width=120)

# PERFECT CENTER AT TOP
title.place(relx=0.5, y=5, anchor='n')

# Admin and User Buttons
font1 = ('times', 15, 'bold')
tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=1, bg='steelblue').place(x=100, y=100)
tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=1, bg='steelblue').place(x=400, y=100)
tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='steelblue').place(x=700, y=100)
tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=1, bg='steelblue').place(x=1000, y=100)

ff = ('times', 12, 'bold')

Button1 = Button(main, text="Exit", command=close,bg='light pink')
Button1.place(x=1100,y=600)
Button1.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=70)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=1,y=150)
text.config(font=font1)

main.config(bg='DarkSlateGray1')
main.mainloop()
