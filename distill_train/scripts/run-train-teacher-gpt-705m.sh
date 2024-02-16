cd ..

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 pretrain-teacher-model.py --config ./config/gpt-705M.yaml
