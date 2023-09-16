import argparse
import json
from tqdm import tqdm
from transformers import RobertaTokenizer
import random
from copy import deepcopy

base_dir = "../MWZProcessor/data_processed"
value_prompt = "all the values mentioned in this turn are {}"
value_prompt_none = "there are no values mentioned in this turn"


def create_negative(cur_vals, past_vals):
    # 正确的样例
    lst = [{"values": cur_vals, "label": 2}]
    # 如果本身的value是空的话
    if len(cur_vals) == 0:
        # 如果之前的value不是空的话随机加入其中的一个value
        for i in range(1, min(len(past_vals) + 1, 3)):
            lst.append({
                "values": random.sample(past_vals, i),
                "label": 0
            })
    else:
        # 从当前的数据中随机删去n个数据
        for i in range(len(cur_vals)):
            new_vals = deepcopy(cur_vals)
            del new_vals[i]
            lst.append({
                "values": new_vals,
                "label": 1
            })
        for i in range(2, len(cur_vals)):
            lst.append({
                "values": random.sample(cur_vals, i),
                "label": 1
            })
        if len(past_vals) != 0:
            for i in range(1, min(len(past_vals) + 1, 3)):
                lst.append({
                    "values": cur_vals + random.sample(past_vals, i),
                    "label": 0
                })
    return lst


def raw_process(mwz_ver, stage):
    with open("{}/{}/{}_raw.json".format(base_dir, mwz_ver, stage), encoding="utf-8") as f:
        data = json.load(f)
    lst = []
    pre_values = []
    for sample in tqdm(data):
        if sample["turn_idx"] == 0:
            pre_values = []
        values = []
        for s, v in sample["turn_label"].items():
            slot, value = s, v
            if value =="dontcare" or value == "none":
                continue
            if "internet" in slot:
                if value == "yes":
                    value = "wifi"
            if "parking" in slot:
                if value == "yes":
                    value = "parking"
            values.append(value)
        extended_sample = create_negative(values, pre_values)
        for each in extended_sample:
            lst.append({
                "dialogue_idx": sample["dialogue_idx"],
                "turn_idx": sample["turn_idx"],
                "user": sample["user"],
                "system": sample["system"],
                "history": sample["history"],
                "values": each["values"],
                "label": each["label"]
            })
        pre_values.extend(values)
    print("Num of {} samples: {}".format(stage, len(lst)))
    with open("../data/filter_data/{}/{}_raw.json".format(mwz_ver, stage), 'w', encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


def process(mwz_ver, stage):
    tokenizer = RobertaTokenizer.from_pretrained(r"roberta-base")
    with open("../data/filter_data/{}/{}_raw.json".format(mwz_ver, stage), encoding="utf-8") as f:
        data = json.load(f)
    lst = []
    # lst_len = []
    for sample in tqdm(data):
        history = ' ; '.join(sample["history"]) + " . " + sample["system"] + ' ; ' + sample["user"]
        history_tokenized = tokenizer.encode(history)[-450:]
        history_tokenized[0] = tokenizer.cls_token_id
        if len(sample["values"]) == 0:
            prompt = value_prompt_none
        else:
            prompt = value_prompt.format(','.join(sample["values"]))
        prompt_tokenized = tokenizer.encode(prompt)[1:]
        input_ids = history_tokenized + prompt_tokenized
        type_ids = [0 for _ in range(len(history_tokenized))] + [1 for _ in range(len(prompt_tokenized))]
        lst.append({
            "flag": sample["dialogue_idx"] + '-' + str(sample["turn_idx"]),
            "input_ids": input_ids,
            "type_ids": type_ids,
            "label": sample["label"]
        })
        # lst_len.append(len(input_ids))
    # print(max(lst_len))
    with open("../data/filter_data/{}/{}.json".format(mwz_ver, stage), 'w', encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


def create_data_from_val_gen(mwz_ver, args):
    tokenizer = RobertaTokenizer.from_pretrained(r"roberta-base")
    with open(args.gen_file_path, encoding="utf-8") as f:
        predict = json.load(f)
    with open("{}/{}/train_leave_raw.json".format(base_dir, mwz_ver), encoding="utf-8") as f:
        raw = json.load(f)
    print(len(raw))
    print(len(predict))
    lst = []
    for idx, each in enumerate(tqdm(predict)):
        history = ' ; '.join(raw[idx]["history"]) + " . " + raw[idx]["system"] + ' ; ' + raw[idx]["user"]
        history_tokenized = tokenizer.encode(history)[-450:]
        history_tokenized[0] = tokenizer.cls_token_id
        # lst_predict = set(each["predict"].split(' | ')) - {' ', ''}
        # lst_ground_truth = set(each["values"])
        # label = 0
        # if lst_predict == lst_ground_truth:
        #     label = 1
        lst_predict = []
        for val in each["predict"].split(' | '):
            if val != ' ' and val != '':
                lst_predict.append(val)
        if len(lst_predict) == 0:
            prompt = value_prompt_none
        else:
            prompt = value_prompt.format(','.join(list(lst_predict)))
        prompt_tokenized = tokenizer.encode(prompt)[1:]
        input_ids = history_tokenized + prompt_tokenized
        type_ids = [0 for _ in range(len(history_tokenized))] + [1 for _ in range(len(prompt_tokenized))]
        lst.append({
            "input_ids": input_ids,
            "type_ids": type_ids,
            "label": 1,
        })
    with open("../data/filter_data/{}/{}.json".format(mwz_ver, args.output_file_name), 'w', encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    mwz_ver = "mwz2_1"
    stages = ["train", "dev", "test"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file_path", type=str, default="")
    parser.add_argument("--output_file_name", type=str, default="")
    args = parser.parse_args()
    if args.gen_file_path == "":
        for stage in stages:
            raw_process(mwz_ver, stage)
            process(mwz_ver, stage)
    else:
        create_data_from_val_gen(mwz_ver, args)
