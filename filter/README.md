## State Value Estimator

### First synthetically generate the dataset by negative sampling

    python data_process.py

### Then train the model:

    python3 main.py \
    --train_batch_size 2 \
    --grad_batches 4 \
    --lr 2e-5 \
    --n_epochs 15 \
    --pretrain roberta-base \
    --name <your_name> \
    --monitor f1 \
    > log.txt
and the path of the best model checkpoint will be stored to './best_model_your_name.txt'.

Notably, you have to change the No. of training samples by modifying the number in 'roberta_nli.py line 105'
### Scoring

You should first generate the dataset using the state value generator's prediction on the remainder of the training data.

    python data_process.py \ 
    --output_file_name level_<ST-iter.> \
    --gen_file_path ../val/generate/level_<ST-iter.> \

We recommend to use the naming rule by setting names of the two models to the same. Or you have to modify the code in '../val/prepare_data.py'

After generating the dataset, you can run:

    python3 generate.py \
    --model_path <best_model_path> \
    --input_file_name level_<ST-iter.> \
    --output_file_name level_<ST-iter.> \
    > log-gen.txt