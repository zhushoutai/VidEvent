#!/bin/bash

echo "start training"
CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/videoevent_slowfast.yaml --output pretrained
echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/videoevent_slowfast.yaml ckpt/videoevent_slowfast_pretrained/epoch_014.pth.tar

