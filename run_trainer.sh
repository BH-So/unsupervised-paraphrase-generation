GPU_ID=$1
TAG=$2
LENGTH=$3  # 32
BATCH_SIZE=$4 # 32
LR=$5  # "1e-6"
ACCUM=$6  # 4
RANDOM_SEED=1234

# TRAIN_DATA="./data/QQP_split/train_sample_preprocessed.txt"
TRAIN_DATA="./data/QQP_split/train_preprocessed.txt"
DEV_DATA="./data/QQP_split/dev_preprocessed.txt"

#    --num_epochs 1 \
#    --save_steps 3 \
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_through_trainer.py \
    --train_data_path ${TRAIN_DATA}\
    --dev_data_path ${DEV_DATA} \
    --max_length ${LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation ${ACCUM} \
    --num_epochs 8 \
    --save_steps 237 \
    --learning_rate ${LR} \
    --tag ${TAG} \
    --seed ${RANDOM_SEED} #--toy --debug
