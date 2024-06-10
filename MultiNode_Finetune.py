# Databricks notebook source
# MAGIC %md
# MAGIC # Multi-node Finetuning

# COMMAND ----------

# MAGIC %pip install fastcore peft==0.11.1 bitsandbytes==0.43.1 mlflow==2.13.1 psutil pynvml trl==0.9.4
# MAGIC %restart_python

# COMMAND ----------

import os
scope_name = 'finetuning_dev'
key_name = 'hf_key'
huggingface_key = dbutils.secrets.get(scope=scope_name, key=key_name)
os.environ['HUGGING_FACE_HUB_TOKEN'] = huggingface_key

# COMMAND ----------

# Databricks configuration and MLflow setup
browser_host = spark.conf.get("spark.databricks.workspaceUrl")
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Make sure the experiment is created first
mlflow_experiment_id = 1576479189931733

os.environ['DATABRICKS_HOST'] = db_host
os.environ['DATABRICKS_TOKEN'] = db_token

# COMMAND ----------

# MAGIC %md 
# MAGIC In order to make the fsdp_qlora framework work with the distributed databricks setup. \
# MAGIC We need to make a few changes to the execution loops and also add mlflow support
# MAGIC
# MAGIC You should clone this fork: https://github.com/Data-drone/fsdp_qlora.git
# MAGIC
# MAGIC Then copy this notebook into that folder
# MAGIC
# MAGIC For the execution we will use the TorchDistributor framework
# MAGIC See here: https://www.databricks.com/blog/2023/04/20/pytorch-databricks-introducing-spark-pytorch-distributor.html

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

num_gpus_per_node = 1
num_nodes = 2
num_processes = num_gpus_per_node * num_nodes
local_status = True if num_nodes == 1 else False

distributor = TorchDistributor(num_processes=num_processes, 
                                local_mode=local_status, use_gpu=True)

distributor.run('train.py', f'--world_size={num_processes}', 
                '--model_name=meta-llama/Meta-Llama-3-8B-Instruct',
                '--train_type=custom_qlora',
                '--sharding_strategy=ddp',
                '--batch_size=4',
                '--log_to=mlflow',
                f'--mlflow_exp_id={mlflow_experiment_id}',
                f'--hugging_face_token={huggingface_key}',
                f'--databricks_host={db_host}',
                f'--databricks_token={db_token}')
# --batch_size 10 \
# --context_length 512 \
# --use_gradient_checkpointing true \
# --use_cpu_offload true \
# --dataset alpaca \
# --reentrant_checkpointing true)
    