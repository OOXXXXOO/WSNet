CUDA_VISIBLE_DEVICES=0 python train.py --backbone xception --lr 0.01 --workers 4 --epochs 200 --batch-size 4 --gpu-ids 0 --checkname deeplab-xception --eval-interval 1 --dataset cityscapes