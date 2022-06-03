export CUDA_VISIBLE_DEVICES=3,4,5
nohup python3 trainimagenet.py > bireal18.out 2>&1 &
# python3 trainimagenet.py | tee -a log/log.txt
