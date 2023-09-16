import pytorch_lightning as pl
import json
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    input_id = [torch.tensor(each["input_ids"], dtype=torch.long) for each in batch]
    type_id = [torch.tensor(each["type_ids"], dtype=torch.long) for each in batch]
    flag = [each["flag"] for each in batch]
    input_ids = pad_sequence(input_id, batch_first=True, padding_value=1)
    type_ids = pad_sequence(type_id, batch_first=True, padding_value=1)
    labels = torch.tensor([each["label"] for each in batch], dtype=torch.long)
    attn_mask = torch.ne(input_ids, 1).long()
    ret = {
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "type_ids": type_ids,
        "labels": labels,
        "flag": flag,
    }
    return ret


class NLIDataSet(pl.LightningDataModule):
    def __init__(self, args):
        super(NLIDataSet, self).__init__()
        self.args = args
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.test_batch_size = args.test_batch_size
        self.n_workers = args.n_workers
        self.train = []
        self.dev = []
        self.test = []

    def setup(self, stage):
        if stage == "fit" or stage is None:
            with open("../data/filter_data/{}/train.json".format(self.args.mwz_ver), encoding="utf-8") as f:
                self.train = json.load(f)
            with open("../data/filter_data/{}/dev.json".format(self.args.mwz_ver), encoding="utf-8") as f:
                self.dev = json.load(f)
        if stage == 'test' or stage is None:
            with open("../data/filter_data/{}/test.json".format(self.args.mwz_ver), encoding="utf-8") as f:
                self.test = json.load(f)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        print(len(self.train))
        return DataLoader(self.train, shuffle=True, batch_size=self.train_batch_size, num_workers=self.n_workers,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev, shuffle=False, batch_size=self.dev_batch_size, num_workers=self.n_workers,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, batch_size=self.test_batch_size, num_workers=self.n_workers,
                          collate_fn=collate_fn)
