import json
import argparse
import torch
from model import T5ValueGenerator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm


def collate_fn(batch):
    input_id = [torch.tensor(each["input_id"], dtype=torch.long) for each in batch]
    label = [torch.tensor(each["label"], dtype=torch.long) for each in batch]
    flag = [each["flag"] for each in batch]
    values = [each["values"] for each in batch]
    input_ids = pad_sequence(input_id, batch_first=True)
    labels = pad_sequence(label, batch_first=True, padding_value=-100)
    attn_mask = torch.ne(input_ids, 0).long()
    ret = {
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "labels": labels,
        "flag": flag,
        "values": values,
    }
    return ret


def generate(args):
    device = torch.device("cuda:0")
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain)
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain)
    # model = T5ValueGenerator(args, pretrain_model, tokenizer).to(device)
    model = T5ValueGenerator.load_from_checkpoint(
        args.model_path,
        args=args, model=pretrain_model, tokenizer=tokenizer
    ).to(device)
    with open("../data/val_data/{}/train_leave.json".format(args.mwz_ver), encoding="utf-8") as f:
        data = json.load(f)
    dataloader = DataLoader(data, shuffle=False, num_workers=args.n_workers, batch_size=args.batch_size, collate_fn=collate_fn)
    lst = []
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        predict = model.my_generate(input_ids, attn_mask)
        for idx, each in enumerate(predict):
            lst.append({
                "flag": batch["flag"][idx],
                "values": batch["values"][idx],
                "predict": each,
            })
    with open("./generate/{}.json".format(args.file_name), 'w', encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pretrain", type=str, default=r"t5-large")
    parser.add_argument("--mwz_ver", type=str, default=r"mwz2_1")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--file_name", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()
    generate(args)
