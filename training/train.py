import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import dotsi

import torch

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from modeling import ViPE


# --------------------------------
#  CONFIG
# --------------------------------
model_name = "gpt2-medium" # Backbone model to use [gpt2-medium, gpt2-large, gpt2, ...]"

device = "cpu"
warmup_steps = 50
max_epochs = 10
learning_rate = 5e-5
batch_size = 2

checkpoint_dir = "/home/branch/misc/lave-test" # Root directory to save everything

train_dataset_path = "../../lavender-extraction/outputs/lave-v0.3/train_scenes.tsv"
eval_dataset_path = "../../lavender-extraction/outputs/lave-v0.3/eval_scenes.tsv"

# Put every config params as dot dict
config = dotsi.Dict(dict())

config.model_name = model_name
config.batch_size = batch_size
config.learning_rate = learning_rate
config.device = device
config.warmup_steps = warmup_steps
config.checkpoint_dir = checkpoint_dir

config.dataset_type = "concept_object"
config.train_dataset_path = train_dataset_path
config.eval_dataset_path = eval_dataset_path
config.concept_list = ['contentment', 'fear', 'excitement', 'amusement', 'disgust', 'something else', 'sadness', 'anger', 'awe']

# ViPE's original params
config.data_dir = None
config.context_length = None

# --------------------------------
#  TRAINING
# --------------------------------

# Create checkpoint directory
os.makedirs(checkpoint_dir, exist_ok=True)

# Save config
with open(os.path.join(checkpoint_dir, "config.json"), 'w') as file:
    file.write(json.dumps(config, ensure_ascii=False, indent=4))

# Set up callbacks
tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(checkpoint_dir, "logs"), name=model_name)
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=5, monitor="val_loss", save_weights_only=True, filename=model_name)
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)

# Set up model
model = ViPE(config)
_ = model.to(config.device)

# Set up trainer
trainer = Trainer(
    accelerator='cpu',
    devices=1,
    callbacks=[checkpoint_callback, early_stop],
    logger=tb_logger,
    max_epochs=max_epochs,
)
# Train
trainer.fit(model)