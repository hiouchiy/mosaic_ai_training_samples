# Databricks notebook source
# MAGIC %md
# MAGIC This is a sample for submitting a model training job to Mosaic Model Training (MCT) from the Databricks Notebook.

# COMMAND ----------

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

# MAGIC %%writefile mosaic_gpt_neox_test.yaml
# MAGIC name: neox-multi-node
# MAGIC image: hiouchiy/gpt-neox:20240909_00 # Docker image provided by EleutherAI
# MAGIC
# MAGIC compute:
# MAGIC   cluster: YOUR_CLUSTER_NAME
# MAGIC   gpus: 8
# MAGIC
# MAGIC
# MAGIC integrations:
# MAGIC - integration_type: git_repo
# MAGIC   git_repo: hiouchiy/gpt-neox
# MAGIC   path: /workspace/gpt-neox
# MAGIC   ssh_clone: false
# MAGIC - integration_type: git_repo
# MAGIC   git_repo: hiouchiy/DeeperSpeed
# MAGIC   path: /workspace/DeeperSpeed
# MAGIC   ssh_clone: false
# MAGIC
# MAGIC command: |
# MAGIC   # Install the requirements for GPT-NeoX
# MAGIC   cd /workspace/gpt-neox
# MAGIC   pip install -r requirements/requirements.txt
# MAGIC
# MAGIC   # Install EleutherAI's fork of deepspeed
# MAGIC   cd /workspace/DeeperSpeed
# MAGIC   pip install .
# MAGIC
# MAGIC   # download and prepare data
# MAGIC   # see https://github.com/EleutherAI/gpt-neox/blob/72c80715c366cc4ad623050d6bcb984fe6638814/README.md?plain=1#L122)
# MAGIC   # for more details on the command
# MAGIC   cd /workspace/gpt-neox
# MAGIC   python prepare_data.py enwik8 -d ./data
# MAGIC
# MAGIC   if [ $NUM_NODES -ge 2 ]; then
# MAGIC     # create a fake hostfile so that GPT-NeoX and DeepSpeed understand the cluster shape
# MAGIC     # Note: this assumes that all nodes have the same number of devices
# MAGIC     python -c '
# MAGIC   import os; \
# MAGIC   import torch; \
# MAGIC   filehandle = open("/tmp/deepspeed_mvapich_hostfile", "w"); \
# MAGIC   world_size = os.environ["WORLD_SIZE"]; \
# MAGIC   device_count = torch.cuda.device_count(); \
# MAGIC   num_nodes = int(world_size) // device_count; \
# MAGIC   _ = [filehandle.write(f"node-{node} slots={device_count}\n") for node in range(num_nodes)]; \
# MAGIC   filehandle.close(); \
# MAGIC     '
# MAGIC     # ls -la /tmp
# MAGIC     cat /tmp/deepspeed_mvapich_hostfile
# MAGIC
# MAGIC     # create a GPT-NeoX config file for data paths, eval split, wandb setup, and launcher
# MAGIC     cd /workspace/gpt-neox/configs
# MAGIC     python -c '
# MAGIC   import json; \
# MAGIC   import os; \
# MAGIC   filehandle = open("extra-configs.yml", "w"); \
# MAGIC   values = { \
# MAGIC     "data-path": "data/enwik8/enwik8_text_document", \
# MAGIC     "use_shared_fs": False, \
# MAGIC     "vocab-file": "data/gpt2-vocab.json", \
# MAGIC     "merge-file": "data/gpt2-merges.txt", \
# MAGIC     "eval-interval": 100, \
# MAGIC     "eval-iters": 100, \
# MAGIC     "split": "949,50,1", \
# MAGIC     "use_wandb": False, \
# MAGIC     "launcher": "mosaicml" \
# MAGIC   }; \
# MAGIC   json.dump(values, filehandle); \
# MAGIC   filehandle.close(); \
# MAGIC     '
# MAGIC
# MAGIC     # run training
# MAGIC     # see https://github.com/EleutherAI/gpt-neox/blob/72c80715c366cc4ad623050d6bcb984fe6638814/README.md?plain=1#L216)
# MAGIC     # for more details on the command
# MAGIC     # see https://github.com/EleutherAI/gpt-neox/blob/72c80715c366cc4ad623050d6bcb984fe6638814/README.md?plain=1#L112
# MAGIC     # for more details on configuration
# MAGIC     cd /workspace/gpt-neox
# MAGIC     ./deepy.py train.py configs/125M-json.yml configs/extra-configs.yml --hostfile /tmp/deepspeed_mvapich_hostfile
# MAGIC   else
# MAGIC     # create a GPT-NeoX config file for data paths, eval split, wandb setup, and launcher
# MAGIC     cd /workspace/gpt-neox/configs
# MAGIC     python -c '
# MAGIC   import json; \
# MAGIC   import os; \
# MAGIC   filehandle = open("extra-configs.yml", "w"); \
# MAGIC   values = { \
# MAGIC     "data-path": "data/enwik8/enwik8_text_document", \
# MAGIC     "vocab-file": "data/gpt2-vocab.json", \
# MAGIC     "merge-file": "data/gpt2-merges.txt", \
# MAGIC     "eval-interval": 100, \
# MAGIC     "eval-iters": 100, \
# MAGIC     "split": "949,50,1", \
# MAGIC     "use_wandb": False \
# MAGIC   }; \
# MAGIC   json.dump(values, filehandle); \
# MAGIC   filehandle.close(); \
# MAGIC     '
# MAGIC
# MAGIC     # run training
# MAGIC     # see https://github.com/EleutherAI/gpt-neox/blob/72c80715c366cc4ad623050d6bcb984fe6638814/README.md?plain=1#L216)
# MAGIC     # for more details on the command
# MAGIC     # see https://github.com/EleutherAI/gpt-neox/blob/72c80715c366cc4ad623050d6bcb984fe6638814/README.md?plain=1#L112
# MAGIC     # for more details on configuration
# MAGIC     cd /workspace/gpt-neox
# MAGIC     ./deepy.py train.py configs/125M-json.yml configs/extra-configs.yml
# MAGIC   fi

# COMMAND ----------

# MAGIC %md
# MAGIC ### (OLD)Single Node (8 GPUs) version
# MAGIC
# MAGIC This is deprecated.

# COMMAND ----------

# MAGIC %%writefile mosaic_gpt_neox_test.yaml
# MAGIC name: neox-single-node
# MAGIC image: shivanshupurohit/gpt-neox:112 # Docker image provided by EleutherAI
# MAGIC
# MAGIC compute:
# MAGIC   cluster: FILL_IN_YOUR_VALUE
# MAGIC   gpus: 8
# MAGIC
# MAGIC
# MAGIC integrations:
# MAGIC - integration_type: git_repo
# MAGIC   git_repo: EleutherAI/gpt-neox
# MAGIC   git_commit: 72c80715c366cc4ad623050d6bcb984fe6638814 # main as of 02-27-2023
# MAGIC   path: /workspace/gpt-neox
# MAGIC   ssh_clone: false
# MAGIC - integration_type: git_repo
# MAGIC   git_repo: EleutherAI/DeeperSpeed
# MAGIC   git_commit: 7069d10d2c9abac50576c84cb7e45910fafa218c # main as of 02-27-2023
# MAGIC   path: /workspace/DeeperSpeed
# MAGIC   ssh_clone: false
# MAGIC
# MAGIC command: |
# MAGIC   # Install the requirements for GPT-NeoX
# MAGIC   cd /workspace/gpt-neox
# MAGIC   pip install -r requirements/requirements.txt
# MAGIC
# MAGIC   # Adopt a patch and install EleutherAI's fork of deepspeed
# MAGIC   cd /workspace/DeeperSpeed
# MAGIC   wget https://raw.githubusercontent.com/hiouchiy/mosaic_ai_training_samples/main/patch/runner.py -O ./deepspeed/launcher/runner.py
# MAGIC   pip install .
# MAGIC
# MAGIC   # create a GPT-NeoX config file for data paths, eval split, wandb setup, and launcher
# MAGIC   cd /workspace/gpt-neox/configs
# MAGIC   python -c '
# MAGIC   import json; \
# MAGIC   import os; \
# MAGIC   filehandle = open("extra-configs.yml", "w"); \
# MAGIC   values = { \
# MAGIC     "data-path": "data/enwik8/enwik8_text_document", \
# MAGIC     "vocab-file": "data/gpt2-vocab.json", \
# MAGIC     "merge-file": "data/gpt2-merges.txt", \
# MAGIC     "eval-interval": 100, \
# MAGIC     "eval-iters": 100, \
# MAGIC     "split": "949,50,1", \
# MAGIC     "use_wandb": False, \
# MAGIC   }; \
# MAGIC   json.dump(values, filehandle); \
# MAGIC   filehandle.close(); \
# MAGIC   '
# MAGIC
# MAGIC   cd /workspace/gpt-neox
# MAGIC
# MAGIC   # download and prepare data
# MAGIC   # see https://github.com/EleutherAI/gpt-neox/blob/72c80715c366cc4ad623050d6bcb984fe6638814/README.md?plain=1#L122)
# MAGIC   # for more details on the command
# MAGIC   python prepare_data.py enwik8 -d ./data
# MAGIC
# MAGIC   pip list
# MAGIC
# MAGIC   # run training
# MAGIC   # see https://github.com/EleutherAI/gpt-neox/blob/72c80715c366cc4ad623050d6bcb984fe6638814/README.md?plain=1#L216)
# MAGIC   # for more details on the command
# MAGIC   # see https://github.com/EleutherAI/gpt-neox/blob/72c80715c366cc4ad623050d6bcb984fe6638814/README.md?plain=1#L112
# MAGIC   # for more details on configuration
# MAGIC   ./deepy.py train.py configs/125M-json.yml configs/extra-configs.yml

# COMMAND ----------

# MAGIC %md
# MAGIC ### (OLD)Multi Node (16 GPUs) version
# MAGIC
# MAGIC This is deprecated.

# COMMAND ----------

# MAGIC %%writefile mosaic_gpt_neox_test.yaml
# MAGIC name: gpt-neox-multi-node
# MAGIC image: shivanshupurohit/gpt-neox:112 # Docker image provided by EleutherAI
# MAGIC
# MAGIC compute:
# MAGIC   cluster: FILL_IN_YOUR_VALUE
# MAGIC   gpus: 16
# MAGIC
# MAGIC
# MAGIC integrations:
# MAGIC - integration_type: git_repo
# MAGIC   git_repo: EleutherAI/gpt-neox
# MAGIC   git_commit: 72c80715c366cc4ad623050d6bcb984fe6638814 # main as of 02-27-2023
# MAGIC   path: /workspace/gpt-neox
# MAGIC   ssh_clone: false
# MAGIC - integration_type: git_repo
# MAGIC   git_repo: EleutherAI/DeeperSpeed
# MAGIC   git_commit: 7069d10d2c9abac50576c84cb7e45910fafa218c # main as of 02-27-2023
# MAGIC   path: /workspace/DeeperSpeed
# MAGIC   ssh_clone: false
# MAGIC
# MAGIC command: |
# MAGIC   # Install the requirements for GPT-NeoX
# MAGIC   cd /workspace/gpt-neox
# MAGIC   pip install -r requirements/requirements.txt
# MAGIC
# MAGIC   # install EleutherAI's fork of deepspeed
# MAGIC   cd /workspace/DeeperSpeed
# MAGIC   pip install .
# MAGIC
# MAGIC   # create a fake hostfile so that GPT-NeoX and DeepSpeed understand the cluster shape
# MAGIC   # Note: this assumes that all nodes have the same number of devices
# MAGIC   python -c '
# MAGIC   import os; \
# MAGIC   import torch; \
# MAGIC   filehandle = open("/tmp/deepspeed_mvapich_hostfile", "w"); \
# MAGIC   world_size = os.environ["WORLD_SIZE"]; \
# MAGIC   device_count = torch.cuda.device_count(); \
# MAGIC   num_nodes = int(world_size) // device_count; \
# MAGIC   _ = [filehandle.write(f"node-{node} slots={device_count}\n") for node in range(num_nodes)]; \
# MAGIC   filehandle.close(); \
# MAGIC   '
# MAGIC   # ls -la /tmp
# MAGIC   cat /tmp/deepspeed_mvapich_hostfile
# MAGIC
# MAGIC   # create a GPT-NeoX config file for data paths, eval split, wandb setup, and launcher
# MAGIC   cd /workspace/gpt-neox/configs
# MAGIC   python -c '
# MAGIC   import json; \
# MAGIC   import os; \
# MAGIC   filehandle = open("extra-configs.yml", "w"); \
# MAGIC   values = { \
# MAGIC     "data-path": "data/enwik8/enwik8_text_document", \
# MAGIC     "use_shared_fs": False, \
# MAGIC     "vocab-file": "data/gpt2-vocab.json", \
# MAGIC     "merge-file": "data/gpt2-merges.txt", \
# MAGIC     "eval-interval": 100, \
# MAGIC     "eval-iters": 100, \
# MAGIC     "split": "949,50,1", \
# MAGIC     "use_wandb": False, \
# MAGIC     "launcher": "mosaicml" \
# MAGIC   }; \
# MAGIC   json.dump(values, filehandle); \
# MAGIC   filehandle.close(); \
# MAGIC   '
# MAGIC
# MAGIC   cd /workspace/gpt-neox
# MAGIC
# MAGIC   # download and prepare data
# MAGIC   # see https://github.com/EleutherAI/gpt-neox/blob/72c80715c366cc4ad623050d6bcb984fe6638814/README.md?plain=1#L122)
# MAGIC   # for more details on the command
# MAGIC   python prepare_data.py enwik8 -d ./data
# MAGIC
# MAGIC   # run training
# MAGIC   # see https://github.com/EleutherAI/gpt-neox/blob/72c80715c366cc4ad623050d6bcb984fe6638814/README.md?plain=1#L216)
# MAGIC   # for more details on the command
# MAGIC   # see https://github.com/EleutherAI/gpt-neox/blob/72c80715c366cc4ad623050d6bcb984fe6638814/README.md?plain=1#L112
# MAGIC   # for more details on configuration
# MAGIC   ./deepy.py train.py configs/125M-json.yml configs/extra-configs.yml --hostfile /tmp/deepspeed_mvapich_hostfile
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch a RUN (=a Training Job)

# COMMAND ----------

from mcli.api.runs import RunConfig, create_run

run_config = RunConfig.from_file('mosaic_gpt_neox_test.yaml')
run_config.compute['cluster'] = 'FILL_IN'
run_config.compute['gpus'] = 8

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


