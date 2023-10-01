Randomly select limited labeled data

Run:

    python main.py \
    --seed <random_seed> \
    --data_ratio <data_ratio> \
    # keep the excluded domain utterances or not, default is not.
    --remove_excluded_utterances <0 | 1> \
    --stage <train | dev | test>

We pre-process the data like TRADE. In TRADE, the excluded domain (police and hospital) utterances are not removed from the training set and all the labeled slot-values of these two domains are removed. It is natural for the domain-slot-based model, such as TRADE to keep the utterances without any label since these samples can act as a sample for 'none' generation. But in the value-based method, the retention of these data may cause different impact. In our paper, for seed 10 we choose not to keep them and the other two seeds (20 and 48) the opposite.

You can get the blank rate of the limited labeled data by running:

    python analyze.py