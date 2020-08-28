# Unsupervised Paraphrase Generation

Unsupervised Paraphrase Generation using Pre-trained Language Model ([paper](https://arxiv.org/abs/2006.05477))

We use Huggingface transformers 3.0.2,  pytorch 1.6.0, and python 3.8 and only support GPU version
Please check `environment.yml` for detail.


## How to run

1. Environment setup
    ```
    conda env create -f environment.yml
    conda activate huggingface
    ```

2. Download dataset

    Please check the `data/README.md` for how to download and preprocess dataset

3. Training (Finetune GPT-2)
    ```
    bash run_train.sh {GPU_ID} {TAG}
    ```
    For example,
    ```
    bash run_train.sh "0,1,2,3" "training_with_4_GPUs" 
    ```

4. Evaluation 
    ```
    bash run_eval.sh {GPU_ID} {TAG} {CHECKPOINT_DIR}
    ```
    For example,
    ```
    bash run_eval.sh 0 "training_with_4_GPUs" "checkpoints/gpt2-medium_training_with_4_GPUs_2020-08-28_12:34:56/checkpoint-3000/"
    ```


## Notice

 - I don't reproduce the results yet. Please feel free to comment for reproduction by creating an issue or a pull request.
 - The experiment on QQP dataset is implemented now. Experiments on SST-2 will be added later.
