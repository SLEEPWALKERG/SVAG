import argparse
import json
from tqdm import tqdm
from transformers import T5Tokenizer
import random
import numpy as np
random.seed(42)


def create_isused(mwz_ver):
    with open("../data/val_data/{}/train_leave.json".format(mwz_ver), encoding="utf-8") as f:
        data = json.load(f)
    with open("./is_used.txt", 'w', encoding="utf-8") as f:
        for i in range(len(data)):
            f.write('0' + '\n')


# train_domain = ["hotel", "attraction", "train"]
prefix = "get informative values: "
history_flag = "<extra_id_0>"
turn_flag = "<extra_id_1>"
prefix_get_intend = "get the requests that the user confirmed or mentioned in this turn"
max_len = 512
base_dir = "../MWZProcessor/data_processed"


def process(args, stage):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain)
    history_flag_id = tokenizer.encode(history_flag)[0]
    prefix_encoded = tokenizer.encode(prefix_get_intend)
    # len_prefix = len(prefix_encoded)
    with open("{}/{}/{}_raw.json".format(base_dir, args.mwz_ver, stage), encoding="utf-8") as f:
        data = json.load(f)
    lst = []
    for sample in tqdm(data):
        values = []
        for s, v in sample["turn_label"].items():
            slot, value = s, v
            if "internet" in slot:
                value = "wifi"
            if "parking" in slot:
                if value == "yes":
                    value = "parking"
            values.append(value)
        history = history_flag + ';'.join(sample["history"]) + turn_flag + sample["system"] + ';' + sample["user"]
        history_tokenized = tokenizer.encode(history)[-384:]
        history_tokenized[0] = history_flag_id
        label = tokenizer.encode(" | ".join(values))
        lst.append({
            "flag": sample["dialogue_idx"] + '-' + str(sample["turn_idx"]),
            "input_id": prefix_encoded + history_tokenized,
            "label": label,
            "values": values,
        })
    with open("../data/val_data/{}/{}.json".format(args.mwz_ver, stage), 'w', encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


def control_blank(lst):
    control_percent = 0.35
    lst_blank = []
    lst_not_blank = []
    for each in lst:
        if len(each["values"]) == 0:
            lst_blank.append(each)
        else:
            lst_not_blank.append(each)
    if len(lst_blank) / (len(lst_not_blank) + len(lst_blank)) + 0.01 < control_percent:
        return lst
    print(len(lst_blank))
    print(len(lst_not_blank))
    random.shuffle(lst_blank)
    idx = int((control_percent * len(lst_not_blank)) / (1 - control_percent) + 1)
    ret = lst_not_blank + lst_blank[:idx]
    random.shuffle(ret)
    return ret


def update_used(is_used, lst):
    for each in lst:
        is_used[each["idx"]] = 1
    return None


def create_data_from_leave(args):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain)
    history_flag_id = tokenizer.encode(history_flag)[0]
    prefix_encoded = tokenizer.encode(prefix_get_intend)
    with open("./generate/{}.json".format(args.level), encoding="utf-8") as f:
        predict = json.load(f)
    with open("{}/{}/train_leave_raw.json".format(base_dir, args.mwz_ver), encoding="utf-8") as f:
        raw = json.load(f)
    with open("../filter/generate/{}.json".format(args.level), encoding="utf-8") as f:
        score = json.load(f)
    with open("./is_used.txt", encoding="utf-8") as f:
        is_used = [int(each) for each in f]
    assert len(predict) == len(is_used)
    lst = []
    for idx, each in enumerate(predict):
        if score[idx] <= args.threshold:
            continue
        if is_used[idx] == 1:
            continue
        values = []
        for val in each["predict"].split(' | '):
            if val != ' ' and val != '':
                values.append(val)
        if len(values) > args.len_ctrl:
            continue
        lst.append({
            "flag": each["flag"],
            "history": raw[idx]["history"],
            "user": raw[idx]["user"],
            "system": raw[idx]["system"],
            "values": values,
            "idx": idx
        })
    lst = control_blank(lst)
    update_used(is_used, lst)
    with open("./is_used.txt", 'w', encoding="utf-8") as f:
        for each in is_used:
            f.write(str(each) + '\n')
    lst_tokenized = []
    s_blank = 0
    for sample in tqdm(lst):
        if len(sample["values"]) == 0:
            s_blank += 1
        history = history_flag + ';'.join(sample["history"]) + turn_flag + sample["system"] + ';' + sample["user"]
        history_tokenized = tokenizer.encode(history)[-384:]
        history_tokenized[0] = history_flag_id
        label = tokenizer.encode(" | ".join(sample["values"]))
        lst_tokenized.append({
            "flag": sample["flag"],
            "input_id": prefix_encoded + history_tokenized,
            "label": label,
            "values": sample["values"],
        })
    print(len(lst_tokenized))
    print("Blank Rate: {}".format(s_blank / len(lst_tokenized)))
    with open("../data/val_data/{}/train_leave_label_{}.json".format(args.mwz_ver, args.level), 'w', encoding="utf-8") as f:
        json.dump(lst_tokenized, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="", type=str)
    parser.add_argument("--threshold", default=1, type=float)
    parser.add_argument("--pretrain", default=r"t5-large", type=str)
    parser.add_argument("--mwz_ver", default="mwz2_1", type=str)
    parser.add_argument("--len_ctrl", default=10, type=int)
    args = parser.parse_args()
    stages = ["train", "dev", "test", "train_leave"]
    if args.level == "":
        for stage in stages:
            process(args, stage)
        create_isused(args.mwz_ver)
    else:
        create_isused(args.mwz_ver)
        create_data_from_leave(args)