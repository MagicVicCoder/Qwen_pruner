import torch
import os
from datetime import datetime

# --- Global Settings ---
# Corrected the trailing space in the endpoint URL
os.environ["HF_HOME"] = "/data/Vic/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,6"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Logging ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# --- MLLM and Dataset Configuration ---
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "lmms-lab/MME"
DATASET_SPLIT = "test[:1000]" # Using a subset for quick testing, adjust as needed
TRAIN_TEST_SPLIT_RATIO = 0.8

# --- Pruning Settings ---
PRUNING_TARGET_RATIO = 0.5  # Target ratio of tokens to keep after pruning (e.g., 0.5 means prune 50%)
PRUNING_MLP_HIDDEN_DIM = 256 # Hidden dimension for the MLP used to predict pruning scores
PRUNING_MLP_DROPOUT = 0.1   # Dropout rate for the MLP
PRUNING_LEARNING_RATE = 1e-4 # Learning rate for training the MLP pruner
PRUNING_NUM_EPOCHS = 10      # Number of epochs to train the pruner (if applicable)
PRUNING_BATCH_SIZE = 4       # Batch size for processing samples during pruning/evaluation

# --- Evaluation Settings ---
EVAL_MODE = "full"  # 可选值："full", "budget", "none"
EVAL_BATCH_SIZE = 4 # Batch size for evaluation