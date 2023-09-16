import argparse
import json
from tqdm import tqdm
from transformers import T5Tokenizer
import random
random.seed(42)


prefix = "get informative values: "
history_flag = "<extra_id_0>"
turn_flag = "<extra_id_1>"
prefix_get_intend = "get the requests that the user confirmed or mentioned in this turn"
max_len = 512
base_dir = "../MWZProcessor/data_processed"


def create_isused(mwz_ver):
    with open("../data/val_data/{}/train_leave.json".format(mwz_ver), encoding="utf-8") as f:
        data = json.load(f)
    with open("./is_used_random.txt", 'w', encoding="utf-8") as f:
        for i in range(len(data)):
            f.write('0' + '\n')


def update_used(is_used, lst):
    for each in lst:
        is_used[each["idx"]] = 1
    return None


def create_random_data_from_leave(args):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain)
    history_flag_id = tokenizer.encode(history_flag)[0]
    prefix_encoded = tokenizer.encode(prefix_get_intend)
    with open("./generate/{}.json".format(args.level), encoding="utf-8") as f:
        predict = json.load(f)
    with open("{}/{}/train_leave_raw.json".format(base_dir, args.mwz_ver), encoding="utf-8") as f:
        raw = json.load(f)
    with open("./is_used_random.txt", encoding="utf-8") as f:
        is_used = [int(each) for each in f]
    lst = []
    for idx, each in enumerate(predict):
        values = []
        if is_used[idx] == 1:
            continue
        for val in each["predict"].split(' | '):
            if val != ' ' and val != '':
                values.append(val)
        lst.append({
            "flag": each["flag"],
            "history": raw[idx]["history"],
            "user": raw[idx]["user"],
            "system": raw[idx]["system"],
            "values": values,
            "idx": idx
        })
    random.shuffle(lst)
    lst = lst[:args.num]
    update_used(is_used, lst)
    with open("./is_used_random.txt", 'w', encoding="utf-8") as f:
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
    with open("../data/val_data/{}/train_leave_label_random_{}.json".format(args.mwz_ver, args.level), 'w', encoding="utf-8") as f:
        json.dump(lst_tokenized, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="", type=str)
    parser.add_argument("--pretrain", default=r"t5-large", type=str)
    parser.add_argument("--mwz_ver", default="mwz2_1", type=str)
    parser.add_argument("--num", default=0, type=int)
    args = parser.parse_args()
    create_isused(args.mwz_ver)  # Before level 1, create it. After that, just comment out this line of code
    create_random_data_from_leave(args)
