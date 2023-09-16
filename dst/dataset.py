import json
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def collate_fn(batch):
    input_id_prompt = [torch.tensor(each["prompt"]) for each in batch]
    input_id_prompt_inverse = [torch.tensor(each["inverse_prompt"]) for each in batch]
    value_encoded = [torch.tensor(each["value_encoded"]) for each in batch]
    slot_encoded = [torch.tensor(each["slot_encoded"]) for each in batch]
    flag = [each["flag"] for each in batch]
    slot = [each["slot"] for each in batch]
    value = [each["value"] for each in batch]
    input_id_prompt = pad_sequence(input_id_prompt, batch_first=True)
    input_id_prompt_inverse = pad_sequence(input_id_prompt_inverse, batch_first=True)
    attn_mask_prompt = torch.ne(input_id_prompt, 0).long()
    attn_mask_prompt_inverse = torch.ne(input_id_prompt_inverse, 0).long()
    value_encoded = pad_sequence(value_encoded, batch_first=True, padding_value=-100)
    slot_encoded = pad_sequence(slot_encoded, batch_first=True, padding_value=-100)
    ret = {
        "input_id_prompt": input_id_prompt,
        "input_id_prompt_inverse": input_id_prompt_inverse,
        "attn_mask_prompt": attn_mask_prompt,
        "attn_mask_prompt_inverse": attn_mask_prompt_inverse,
        "value_encoded": value_encoded,
        "slot_encoded": slot_encoded,
        "flag": flag,
        "value": value,
        "slot": slot,
    }
    return ret


class DSTDataset(pl.LightningDataModule):
    def __init__(self, args):
        super(DSTDataset, self).__init__()
        self.args = args
        self.train = []
        self.dev = []
        self.test = []

    def setup(self, stage):
        if stage == "fit" or stage is None:
            with open("../data/dst_data/{}/train.json".format(self.args.mwz_ver), encoding="utf-8") as f:
                self.train = json.load(f)
            with open("../data/dst_data/{}/dev.json".format(self.args.mwz_ver), encoding="utf-8") as f:
                self.dev = json.load(f)
        if stage == "test" or stage is None:
            with open("../data/dst_data/{}/{}.json".format(self.args.mwz_ver, self.args.test_name), encoding="utf-8") as f:
                self.test = json.load(f)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=False, batch_size=self.args.train_batch_size,
                          num_workers=self.args.n_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev, shuffle=False, batch_size=self.args.dev_batch_size, num_workers=self.args.n_workers,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, batch_size=self.args.test_batch_size,
                          num_workers=self.args.n_workers, collate_fn=collate_fn)
