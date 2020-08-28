GPU_ID=$1
TAG=$2
CHECKPOINT_DIR=$3

T="1.0"  # temperature
k=10
p="1.0"
N_GEN=10
SEED="1234"

INPUT_FILE="./data/QQP_split/test_input.txt"
PREPROCESSED="./data/QQP_split/test_input_preprocessed.txt"
TARGET="./data/QQP_split/test_target.txt"
FILENAME="inferenced_${TAG}_top-${k}-p${p//./_}-T${T//./_}_seed${SEED}.txt"

CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
    --data_path ${PREPROCESSED} \
    --checkpoint ${CHECKPOINT_DIR} \
    --save "./results/${FILENAME}" \
    --decoding "sampling" \
    --k ${k} \
    --p ${p} \
    --temperature ${T} \
    --num_generate ${N_GEN} \
    --seed ${SEED} \
    --tag ${TAG}

CUDA_VISIBLE_DEVICES=$GPU_ID python postprocessing.py \
    --input ${INPUT_FILE} \
    --paraphrase "./results/${FILENAME}" \
    --output "./results/filtered/${FILENAME}" \
    --model "bert-base-nli-stsb-mean-tokens" \
    --tag ${TAG}

CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py \
    --generated "./results/filtered/${FILENAME}" \
    --ground_truth ${TARGET} \
    --tag ${TAG}
