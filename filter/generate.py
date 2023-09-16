import json
import argparse
import torch
from roberta_nli import RobertaNLI
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm


def collate_fn(batch):
    input_id = [torch.tensor(each["input_ids"], dtype=torch.long) for each in batch]
    type_id = [torch.tensor(each["type_ids"], dtype=torch.long) for each in batch]
    labels = torch.tensor([each["label"] for each in batch], dtype=torch.long)
    input_ids = pad_sequence(input_id, batch_first=True)
    type_ids = pad_sequence(type_id, batch_first=True, padding_value=1)
    attn_mask = torch.ne(input_ids, 0).long()
    ret = {
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "labels": labels,
        "type_ids": type_ids,
    }
    return ret


def generate(args):
    device = torch.device("cuda:0")
    pretrain_model = RobertaForSequenceClassification.from_pretrained(args.pretrain, num_labels=args.num_labels)
    # model = RobertaNLI(args, pretrain_model).to(device)
    model = RobertaNLI.load_from_checkpoint(
        args.model_path,
        args=args, pretrain=pretrain_model
    ).to(device)
    with open("../data/filter_data/{}/{}.json".format(args.mwz_ver, args.input_file_name), encoding="utf-8") as f:
        data = json.load(f)
    dataloader = DataLoader(data, shuffle=False, num_workers=args.n_workers, batch_size=args.batch_size,
                            collate_fn=collate_fn)
    lst = []
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        inp = {
            "input_ids": batch["input_ids"].to(device),
            "attn_mask": batch["attn_mask"].to(device),
            "type_ids": batch["type_ids"].to(device),
            "labels": batch["labels"].to(device)
        }
        acc = model.my_generate(inp, batch_idx)
        lst.extend(acc.tolist())
    with open("./generate/{}.json".format(args.output_file_name), 'w', encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pretrain", type=str, default=r"roberta-base")
    parser.add_argument("--mwz_ver", type=str, default=r"mwz2_1")
    parser.add_argument("--n_workers", type=int, default=2)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--output_file_name", type=str, default="")
    parser.add_argument("--input_file_name", type=str, default="")
    parser.add_argument("--num_labels", type=int, default=3)
    args = parser.parse_args()
    generate(args)
