GPU_ID=$1
CHECKPOINT_DIR=$2
#CHECKPOINT_DIR="./checkpoints/gpt2-medium_Trunc32LR1e-6_2020-08-09_00:52:53/ep9/"
#CHECKPOINT_DIR="./checkpoints/gpt2-medium_length32_LRlower_noSR_2020-08-18_21:34:52/ep6/"

T=$3  # temperature
#temperature_list=("1.0" "1.1" "1.2" "1.4" "1.7" "2.0" "2.4" "2.8")


k=10
p="1.0"
SEED="1234"

#for T in ${temperature_list[@]}; do
INPUT_FILE="./data/QQP_split/ver2/test_input.txt"
PREPROCESSED="./data/QQP_split/ver2/test_input_preprocessed.txt"
FILENAME="inferenced_top-${k}-p${p//./_}-T${T//./_}_seed${SEED}.txt"

# Skip preprocessing of "./data/QQP_split/test_input.txt" <- It has to be run once

#    --data_path ${PREPROCESSED} \
CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
    --data_path ${INPUT_FILE} \
    --checkpoint ${CHECKPOINT_DIR} \
    --save "./results/${FILENAME}" \
    --decoding "sampling" \
    --k ${k} \
    --p ${p} \
    --temperature ${T} \
    --num_generate 10 \
    --seed ${SEED}

CUDA_VISIBLE_DEVICES=$GPU_ID python postprocessing.py \
    --input ${INPUT_FILE} \
    --paraphrase "./results/${FILENAME}" \
    --output "./results/filtered/${FILENAME}"

CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py \
    --generated "./results/filtered/${FILENAME}" \
    --ground_truth "./data/QQP_split/test.txt" \
    --metrics "meteor,rouge"
#done
