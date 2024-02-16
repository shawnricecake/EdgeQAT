
CHECKPOINT_PATH=/path/llama-58m

CUDA_VISIBLE_DEVICES=7 \
python3 eval_zeroshot.py \
        $CHECKPOINT_PATH \
        decoder \
