# Databricks notebook source
# MAGIC %md
# MAGIC ## Install MCLI into this Notebook

# COMMAND ----------

# MAGIC %pip install --upgrade mosaicml-cli

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set MosaicML API Key

# COMMAND ----------

# MAGIC %md
# MAGIC You need to create an API key in the MCT web console and store it in Databricks secrets before executing the following cell.

# COMMAND ----------

import mcli
mcli.set_api_key(dbutils.secrets.get("hiouchiy", "mosaicml_token"))

# COMMAND ----------

# MAGIC %md
# MAGIC Check if the API key is correctly set by getting cluster's information from MCT.

# COMMAND ----------

clusters = mcli.get_clusters()
mcli.get_cluster(clusters[0]) if len(clusters) > 0 else "No clusters found."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a RUN definition file (.yaml) for testing

# COMMAND ----------

# MAGIC %%writefile mosaic_gpt_test.yaml
# MAGIC # 1B LLM training config
# MAGIC name: my-test-run-mpt-1b-llm-foundry
# MAGIC compute:
# MAGIC   cluster: r8z11
# MAGIC   gpus: 16
# MAGIC image: mosaicml/llm-foundry:2.3.0_cu121_flash2-latest
# MAGIC integrations:
# MAGIC   - integration_type: git_repo
# MAGIC     git_repo: hiouchiy/llm-foundry
# MAGIC     git_commit: hiouchiy-test
# MAGIC     pip_install: .[gpu-flash2,databricks]
# MAGIC   - integration_type: mlflow
# MAGIC     tracking_uri: databricks
# MAGIC     experiment_name: /Users/hiroshi.ouchiyama@databricks.com/mct_experiment
# MAGIC command: >-
# MAGIC   cd llm-foundry/scripts
# MAGIC
# MAGIC   composer train/train.py /mnt/config/parameters.yaml
# MAGIC parameters:
# MAGIC   run_name:  # If left blank, will be read from env var $RUN_NAME
# MAGIC
# MAGIC   max_seq_len: 2048
# MAGIC   global_seed: 17
# MAGIC   data_local: ./my-copy-c4
# MAGIC   data_remote: dbfs:/Volumes/shared/hiouchiymct/data
# MAGIC
# MAGIC   # Model
# MAGIC   model:
# MAGIC     name: mpt_causal_lm
# MAGIC     init_device: meta
# MAGIC     d_model: 2048
# MAGIC     n_heads: 16  # Modified 24->16 so that d_head == 128 to satisfy FlashAttention
# MAGIC     n_layers: 24
# MAGIC     expansion_ratio: 4
# MAGIC     max_seq_len: ${max_seq_len}
# MAGIC     vocab_size: 50368
# MAGIC     attn_config:
# MAGIC       attn_impl: flash
# MAGIC
# MAGIC   # Tokenizer
# MAGIC   tokenizer:
# MAGIC     name: EleutherAI/gpt-neox-20b
# MAGIC     kwargs:
# MAGIC       model_max_length: ${max_seq_len}
# MAGIC
# MAGIC   # Dataloaders
# MAGIC   train_loader:
# MAGIC     name: text
# MAGIC     dataset:
# MAGIC       local: ${data_local}
# MAGIC       remote: ${data_remote}
# MAGIC       split: train_small
# MAGIC       shuffle: true
# MAGIC       max_seq_len: ${max_seq_len}
# MAGIC       shuffle_seed: ${global_seed}
# MAGIC     drop_last: true
# MAGIC     num_workers: 8
# MAGIC
# MAGIC   eval_loader:
# MAGIC     name: text
# MAGIC     dataset:
# MAGIC       local: ${data_local}
# MAGIC       remote: ${data_remote}
# MAGIC       split: val_small
# MAGIC       shuffle: false
# MAGIC       max_seq_len: ${max_seq_len}
# MAGIC       shuffle_seed: ${global_seed}
# MAGIC     drop_last: false
# MAGIC     num_workers: 8
# MAGIC
# MAGIC   # Optimization
# MAGIC   scheduler:
# MAGIC     name: cosine_with_warmup
# MAGIC     t_warmup: 100ba
# MAGIC     alpha_f: 0.1
# MAGIC
# MAGIC   optimizer:
# MAGIC     name: decoupled_adamw
# MAGIC     lr: 2.0e-4
# MAGIC     betas:
# MAGIC     - 0.9
# MAGIC     - 0.95
# MAGIC     eps: 1.0e-08
# MAGIC     weight_decay: 0.0
# MAGIC
# MAGIC   algorithms:
# MAGIC     gradient_clipping:
# MAGIC       clipping_type: norm
# MAGIC       clipping_threshold: 1.0
# MAGIC
# MAGIC   max_duration: 30ba
# MAGIC   eval_interval: 5ba
# MAGIC   eval_first: false
# MAGIC   eval_subset_num_batches: -1
# MAGIC   global_train_batch_size: 512
# MAGIC
# MAGIC   # System
# MAGIC   seed: ${global_seed}
# MAGIC   device_eval_batch_size: 4
# MAGIC   device_train_microbatch_size: 4
# MAGIC   # device_train_microbatch_size: auto
# MAGIC   precision: amp_bf16
# MAGIC
# MAGIC   # FSDP
# MAGIC   fsdp_config:
# MAGIC     sharding_strategy: FULL_SHARD
# MAGIC     mixed_precision: PURE
# MAGIC     activation_checkpointing: false
# MAGIC     activation_checkpointing_reentrant: false
# MAGIC     activation_cpu_offload: false
# MAGIC     limit_all_gathers: true
# MAGIC
# MAGIC   # Logging
# MAGIC   progress_bar: false
# MAGIC   log_to_console: true
# MAGIC   console_log_interval: 1ba
# MAGIC
# MAGIC   callbacks:
# MAGIC     speed_monitor:
# MAGIC       window_size: 10
# MAGIC     lr_monitor: {}
# MAGIC     memory_monitor: {}
# MAGIC     runtime_estimator: {}
# MAGIC     hf_checkpointer:
# MAGIC       overwrite: true
# MAGIC       precision: bfloat16
# MAGIC       save_folder: dbfs:/databricks/mlflow-tracking/{mlflow_experiment_id}/{mlflow_run_id}/artifacts/checkpoints
# MAGIC       save_interval: 5ba
# MAGIC       mlflow_logging_config:
# MAGIC         task: llm/v1/chat
# MAGIC         metadata:
# MAGIC           task: llm/v1/chat
# MAGIC       mlflow_registered_model_name: mpt-1b-hiouchiy
# MAGIC
# MAGIC   loggers:
# MAGIC     mlflow:
# MAGIC       tracking_uri: databricks
# MAGIC       model_registry_uri: databricks-uc
# MAGIC       model_registry_prefix: shared.hiouchiymct
# MAGIC
# MAGIC   # loggers:
# MAGIC   #   wandb: {}
# MAGIC
# MAGIC   # Checkpoint to local filesystem or remote object store
# MAGIC   # save_interval: 2000ba
# MAGIC   # save_num_checkpoints_to_keep: 1  # Important, this cleans up checkpoints saved to DISK
# MAGIC   # save_folder: ./{run_name}/checkpoints
# MAGIC   # save_folder: s3://my-bucket/my-folder/{run_name}/checkpoints
# MAGIC   save_interval: 5ba
# MAGIC   save_folder: dbfs:/databricks/mlflow-tracking/{mlflow_experiment_id}/{mlflow_run_id}/artifacts/{run_name}/checkpoints
# MAGIC
# MAGIC   # Load from local filesystem or remote object store
# MAGIC   # load_path: ./gpt-1b/checkpoints/latest-rank{rank}.pt
# MAGIC   # load_path: s3://my-bucket/my-folder/gpt-1b/checkpoints/latest-rank{rank}.pt
# MAGIC   load_path: dbfs:/databricks/mlflow-tracking/1322175094395138/5c4432788793429bb9403ecad312d5f9/artifacts/mpt-1b-quickstart-llm-foundry-DA2Edb/checkpoints/ep0-ba3-rank0.pt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch a RUN (=a Training Job)

# COMMAND ----------

from mcli.api.runs import RunConfig, create_run

run_config = RunConfig.from_file('mosaic_gpt_test.yaml')
created_run = create_run(run_config)
print(f'Started run: {created_run.run_uid} at {created_run.created_at}')
print(f'Cluster: {created_run.cluster}')
print(f'Number of GPUs: {created_run.gpus} ({created_run.gpu_type})')
print(f'Number of Nodes: {created_run.node_count}')
print(f'Cluster: {created_run.cluster}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor logs from the RUN synchronously

# COMMAND ----------

for line in mcli.follow_run_logs(created_run):
    print(line)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor logs from the RUN asynchronously

# COMMAND ----------

for line in mcli.get_run_logs(created_run):
    print(line)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop the RUN

# COMMAND ----------

mcli.stop_run(created_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete the RUN

# COMMAND ----------

mcli.delete_run(created_run)

# COMMAND ----------


