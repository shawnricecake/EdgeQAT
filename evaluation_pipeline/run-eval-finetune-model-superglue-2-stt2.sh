#!/bin/bash

MODEL_PATH=""
TASK_NAME="glue"
SUBTASK_NAME="sst2" # "cola" "sst2" "mrpc" "qqp" "mnli" "mnli-mm" "qnli" "rte" "boolq" "multirc" "wsc"
LR=5e-5         # default: 5e-5
BSZ=64            # default: 64
MAX_EPOCHS=6    # default: 10
PATIENCE=10       # default: 10
EVAL_EVERY=20000    # default: 200
SEED=12           # default: 12

if [[ "$SUBTASK_NAME" = "mnli" ]]; then
    VALID_NAME="validation_matched"
    OUT_DIR="mnli"
elif [[ "$SUBTASK_NAME" = "mnli-mm" ]]; then
    VALID_NAME="validation_mismatched"
    SUBTASK_NAME="mnli"
    OUT_DIR="mnli-mm"
else
    VALID_NAME="validation"
    OUT_DIR=$SUBTASK_NAME
fi

mkdir -p $MODEL_PATH/finetune/$OUT_DIR/

python eval_finetune_classification.py \
  --model_name_or_path $MODEL_PATH \
  --output_dir $MODEL_PATH/finetune/$OUT_DIR/ \
  --train_file /path/$TASK_NAME\_filtered/$SUBTASK_NAME.train.json \
  --validation_file /path/$TASK_NAME\_filtered/$SUBTASK_NAME.$VALID_NAME.json \
  --do_train \
  --do_eval \
  --do_predict \
  --use_fast_tokenizer False \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --learning_rate $LR \
  --num_train_epochs $MAX_EPOCHS \
  --evaluation_strategy steps \
  --patience $PATIENCE \
  --eval_every $EVAL_EVERY \
  --eval_steps $EVAL_EVERY \
  --save_steps $EVAL_EVERY \
  --overwrite_output_dir \
  --seed $SEED
