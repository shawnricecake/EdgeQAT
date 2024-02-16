cd ..

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 pretrain-teacher-model.py --config ./config/llama-360M.yaml

