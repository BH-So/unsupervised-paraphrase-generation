GPU_ID=$1
TAG=$2
LR=$3
LENGTH=$4
BATCH_SIZE=$5
RANDOM_SEED=1234

TRAIN_DATA="./data/QQP_split/train_ver2_trunc${LENGTH}.txt"
DEV_DATA="./data/QQP_split/dev_ver2_trunc${LENGTH}.txt"

#    --train_data_path ./data/QQP_split/train_preparation.txt \
#    --dev_data_path ./data/QQP_split/dev_preparation.txt \
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --train_data_path ${TRAIN_DATA}\
    --dev_data_path ${DEV_DATA} \
    --max_length ${LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --num_epochs 10 \
    --tag ${TAG} \
    --seed ${RANDOM_SEED} # --toy
