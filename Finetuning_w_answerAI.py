# Databricks notebook source
# MAGIC %md
# MAGIC # Finetuning on Databricks
# MAGIC
# MAGIC In this notebook we will leverage: \
# MAGIC https://github.com/AnswerDotAI/fsdp_qlora.git
# MAGIC
# MAGIC You will need to import that repo into databricks as a Git Folder then copy this notebook into the directory \
# MAGIC *Note* At this stage this repo only has wandb integration not mlflow.

# COMMAND ----------

# MAGIC %pip install fastcore peft==0.11.1 bitsandbytes==0.43.1 mlflow==2.13.1 psutil pynvml trl==0.9.4
# MAGIC %restart_python

# COMMAND ----------

# To download llama models from huggingface you need: 
# - create an account
# - ask for assess from meta
# - create a huggingface key to get remote access
# - save key to databricks secrets
# 
# Once it is saved we can retrieve it from secrets and make it available to the node

import os
scope_name = 'finetuning_dev'
key_name = 'hf_key'
huggingface_key = dbutils.secrets.get(scope=scope_name, key=key_name)
os.environ['HUGGING_FACE_HUB_TOKEN'] = huggingface_key

# COMMAND ----------

# MAGIC %md The following code cell assumes that this notebook is in the same folder as the train script 

# COMMAND ----------

!python train.py \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--batch_size 10 \
--context_length 512 \
--precision bf16 \
--train_type qlora \
--use_gradient_checkpointing true \
--use_cpu_offload true \
--dataset alpaca \
--reentrant_checkpointing true