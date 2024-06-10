# Data & AI Summit 2024 Finetuning Guide

This repo is the accompany a talk on finetuning and how to do it.

## Single Node Training on Databricks

A good starting repo for running finetuning is the https://github.com/AnswerDotAI/fsdp_qlora repo.
It has less dependencies than most and also has optimizations to minimise ram usage as well for running on small 24GB VRAM GPUs

See: Finetuning_w_answerAI notebook

## Multi Node Training on Databricks

When we move to multi-node then the easiest path is to use (TorchDistributor)[https://www.databricks.com/blog/2023/04/20/pytorch-databricks-introducing-spark-pytorch-distributor.html] and (DeepSpeedDistributor)[https://community.databricks.com/t5/technical-blog/introducing-the-deepspeed-distributor-on-databricks/ba-p/59641]

These allow us to run code distributed training processes on Spark with minimal changes.
See: the MultiNode_Finetune



