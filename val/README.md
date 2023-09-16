## State Value Generator with Self-Training

### First generate the dataset

    python prepare_data.py

### Then train the model with 

    python main.py \
    --is_train 1 \
    --lr 5e-5 \
    --name <your_name> \
    --grad_batches 1 \
    --result_name result \
    --num_train <No. of training samples> \
    --n_epochs 15 \
    --pretrain t5-large \
    --warm 1000 \
    > log.txt

### After the training process, the model will automatically test on the test set with the best model on validation set. If you want to run a certain checkpoint on the test set, you can run:

    python main.py \
    --pretrain t5-large \
    --result_name <your_name> \
    --is_train 0 \
    --model_path <ckpt_path> \
    > log_gen.txt

### To generate pseudo label for the rest of unlabeled data, you can run:

    python3 generate.py \
    --file_name level_1 \
    --model_path <ckpt_path> \
    > log_gen.txt

### After get the estimator's score on the pseudo-labeled data, you can get the statistical information by:

    python eval_nli_filter.py

### To get the selected pseudo-labeled data, you can run:

    python prepare_data.py
    --level level_1
    --threshold 0.98

### train a new model with the selected pseudo-labeled data and the original golden training data

    python main.py \
    --is_train 1 \
    --lr 5e-5 \
    --name <your_name> \
    --grad_batches 1 \
    --result_name result-1 \
    --num_train <No. of training samples> \
    --level_1 train_leave_label_level_1 \
    --n_epochs 5 \
    --pretrain t5-large \
    --warm 1000 \
    --model_path ckpt \
    > log-1.txt

### Treat the new model as a new teacher, and run multiple iterations till the accuracy do not increase

### Additionally, you can randomly select pseudo-laveled data by:
    python random_data.py --level level_1 --num <No. of training samples>

### Change the config "level_1" in main.py to run model with random pseudo-labeled data and the golden data.

### Evaluation:
    python analyze.py --result_path --result_path <default: ./result/result.json>