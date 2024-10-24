# Databricks notebook source
# MAGIC %md
# MAGIC # MCT用のデータセット作成ノートブック
# MAGIC
# MAGIC 参考：https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md

# COMMAND ----------

# MAGIC %md
# MAGIC ## 準備

# COMMAND ----------

# MAGIC %md
# MAGIC LLM-Foundryのリポジトリをクローン

# COMMAND ----------

# MAGIC %sh git clone https://github.com/mosaicml/llm-foundry.git /tmp/llm-foundry

# COMMAND ----------

# MAGIC %cd /tmp/llm-foundry
# MAGIC %pip install -e .
# MAGIC %pip install --upgrade typing_extensions
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Hugging Faceにログイン（アクセストークン使用）

# COMMAND ----------

from huggingface_hub import login
login()

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks Unity Catalogにカタログ、スキーマ、ボリュームを作成

# COMMAND ----------

catalog = "shared"
schema = "hiouchiymct"
volume = "dataset"

# COMMAND ----------

# カタログがなければ作成する
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")

# スキーマがなければ作成する
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# ボリュームがなければ作成する
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC Linuxコマンドを実行用に必要な値を環境変数に設定

# COMMAND ----------

import os
os.environ['OUTPUT_ROOT'] = f"/Volumes/{catalog}/{schema}/{volume}"
os.environ['TOKENIZER_NAME'] = "meta-llama/Llama-3.2-1B"
os.environ['MAX_SEQ_LEN'] = 4096

# COMMAND ----------

# MAGIC %md
# MAGIC ## 事前学習用データの準備

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1. 元データがJSON形式の場合

# COMMAND ----------

import pandas as pd 

json_path = f'/tmp/llm-foundry/scripts/data_prep/example_data/arxiv.jsonl'

pandasDF = pd.read_json(path_or_buf=json_path, lines=True)
display(pandasDF)

# COMMAND ----------

# MAGIC %sh 
# MAGIC python3 /tmp/llm-foundry/scripts/data_prep/convert_dataset_json.py \
# MAGIC --path /tmp/llm-foundry/scripts/data_prep/example_data/arxiv.jsonl \
# MAGIC --out_root $OUTPUT_ROOT/my-copy-arxiv \
# MAGIC --split train \
# MAGIC --tokenizer $TOKENIZER_NAME \
# MAGIC --eos_text '<|end_of_text|>' \
# MAGIC --compression zstd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2. 元データがTEXTの場合

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir /tmp/shakespeare && cd /tmp/shakespeare
# MAGIC curl -O https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
# MAGIC echo '------------------------------------------------------'
# MAGIC head t8.shakespeare.txt

# COMMAND ----------

# MAGIC %sh
# MAGIC python3 /tmp/llm-foundry/scripts/data_prep/convert_text_to_mds.py \
# MAGIC   --output_folder $OUTPUT_ROOT/my-copy-shakespeare/train \
# MAGIC   --input_folder /tmp/shakespeare \
# MAGIC   --concat_tokens $MAX_SEQ_LEN \
# MAGIC   --tokenizer $TOKENIZER_NAME \
# MAGIC   --use_tokenizer_eos \
# MAGIC   --compression zstd

# COMMAND ----------

# MAGIC %sh
# MAGIC python3 /tmp/llm-foundry/scripts/data_prep/convert_text_to_mds.py \
# MAGIC   --output_folder $OUTPUT_ROOT/my-copy-shakespeare/val \
# MAGIC   --input_folder /tmp/shakespeare \
# MAGIC   --concat_tokens $MAX_SEQ_LEN \
# MAGIC   --tokenizer $TOKENIZER_NAME \
# MAGIC   --use_tokenizer_eos \
# MAGIC   --compression zstd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 3. 元データがHuggingFace Datasetの場合

# COMMAND ----------

# MAGIC %sh
# MAGIC python3 /tmp/llm-foundry/scripts/data_prep/convert_dataset_hf.py \
# MAGIC   --dataset allenai/c4 \
# MAGIC   --data_subset ja \
# MAGIC   --out_root $OUTPUT_ROOT/my-copy-c4-ja \
# MAGIC   --splits train_small val_small \
# MAGIC   --concat_tokens $MAX_SEQ_LEN \
# MAGIC   --tokenizer $TOKENIZER_NAME \
# MAGIC   --eos_text '<|end_of_text|>' \
# MAGIC   --compression zstd

# COMMAND ----------

# MAGIC %md
# MAGIC ## ファインチューニング用データの準備

# COMMAND ----------

# MAGIC %sh
# MAGIC python3 /tmp/llm-foundry/scripts/data_prep/convert_finetuning_dataset.py \
# MAGIC     --dataset kunishou/databricks-dolly-15k-ja \
# MAGIC     --preprocessor "llmfoundry.data.finetuning.tasks:dolly_preprocessing_function" \
# MAGIC     --splits train \
# MAGIC     --out_root $OUTPUT_ROOT/my-copy-dolly-15k-ja
