cd ..

CUDA_VISIBLE_DEVICES=1 \
python3 pretrain-llama-58m
-distill.py
