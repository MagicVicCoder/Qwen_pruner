import torch
import os
from datetime import datetime

# --- Global Settings ---
# Use a cache directory within your project or home directory
os.environ["HF_HOME"] = os.path.expanduser("~/Vic/tokenprune/data/huggingface_cache")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,6"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Logging ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# --- MLLM and Dataset Configuration ---
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# ScreenSpot-Pro bbox 任务
DATASET_NAME = "Voxel51/ScreenSpot-Pro"
DATASET_SPLIT = "train"  # ScreenSpot-Pro 通常只提供 train/validation 等划分
TRAIN_TEST_SPLIT_RATIO = 0.8

# --- Pruning Settings ---
PRUNING_TARGET_RATIO = 0.5  # Target ratio of tokens to keep after pruning (e.g., 0.5 means prune 50%)

# --- MLP Pruner Settings ---
PRUNING_MLP_HIDDEN_DIM = 256 # Hidden dimension for the MLP used to predict pruning scores
PRUNING_MLP_DROPOUT = 0.1   # Dropout rate for the MLP
PRUNING_LEARNING_RATE = 1e-4 # Learning rate for training the MLP pruner (if applicable)
PRUNING_NUM_EPOCHS = 10      # Number of epochs to train the pruner (if applicable)
PRUNING_BATCH_SIZE = 4       # Batch size for processing samples during pruning/evaluation

# --- DivPrune Settings ---
# Weight for balancing importance vs diversity in DivPrune's selection criterion
# Alpha * Importance + (1-Alpha) * Diversity
DIV_PRUNE_IMPORTANCE_WEIGHT = 0.5 # Adjust this value to tune the trade-off

# --- Evaluation Settings ---
EVAL_MODE = "full"  # 可选值："full", "budget", "none"
EVAL_BATCH_SIZE = 4 # Batch size for evaluation
BBOX_SUCCESS_IOU = 0.5  # IoU threshold for considering bbox prediction correct
