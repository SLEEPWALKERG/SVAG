import argparse
import json
from transformers import T5Tokenizer
from tqdm import tqdm


base_dir = "../MWZProcessor/data_processed"


prompt_value = [
    "belief states: slot={}, value=",
    "belief states: {}=",
    "{} is the slot of ",
    "what is the value of {}?"
]


prompt_slot = [
    "belief states: value={}, slot=",
    "belief states: {}=",
    "{} is the value of ",
    "what is the slot type of {}?"
]


prefix = "answer the question:"


def process(stage, args):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain)
    prefix_encoded = tokenizer.encode(prefix)
    with open("./{}/{}/{}_raw.json".format(base_dir, args.mwz_ver, stage), encoding="utf-8") as f:
        data = json.load(f)
    lst = []
    length = []
    for sample in tqdm(data):
        history = ';'.join(sample["history"]) + ';' + sample["system"] + ';' + sample["user"]
        history_tokenized = tokenizer.encode(history)[-480:]
        for s, v in sample["turn_label"].items():
            slot, value = s, v
            if "internet" in slot:
                if value == "yes":
                    value = "wifi"
            if "parking" in slot:
                if value == "yes":
                    value = "parking"
            prompt = tokenizer.encode(prompt_slot[3].format(value))
            prompt_inverse = tokenizer.encode(prompt_value[3].format(slot))
            lst.append({
                "flag": sample["dialogue_idx"] + '-' + str(sample["turn_idx"]),
                "prompt": prefix_encoded + history_tokenized + prompt,
                "inverse_prompt": prefix_encoded + history_tokenized + prompt_inverse,
                "value_encoded": tokenizer.encode(value),
                "slot_encoded": tokenizer.encode(slot),
                "value": value,
                "slot": slot,
            })
            length.append(len(lst[-1]["prompt"]))
    print("Number of {} samples: {}".format(stage, len(lst)))
    with open("../data/dst_data/{}/{}.json".format(args.mwz_ver, stage), 'w', encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


def process_value_gen(args):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain)
    prefix_encoded = tokenizer.encode(prefix)
    dic = {}
    with open(args.value_path, encoding="utf-8") as f:
        val = json.load(f)
    for each in val:
        dic[each["flag"]] = list(set(each["predict"].split(' | ')) - {' ', ''})
    with open("{}/{}/test_raw.json".format(base_dir, args.mwz_ver), encoding="utf-8") as f:
        data = json.load(f)
    lst = []
    for sample in tqdm(data):
        flag = sample["dialogue_idx"] + '-' + str(sample["turn_idx"])
        if flag in dic:
            for v in dic[flag]:
                history = ';'.join(sample["history"]) + ';' + sample["system"] + ';' + sample["user"]
                history_tokenized = tokenizer.encode(history)[-480:]
                prompt = tokenizer.encode(prompt_slot[3].format(v))
                lst.append({
                    "flag": flag,
                    "prompt": prefix_encoded + history_tokenized + prompt,
                    "inverse_prompt": [0],
                    "value_encoded": [0],
                    "slot_encoded": [0],
                    "value": v,
                    "slot": "",
                })
    with open("../data/dst_data/{}/test_gen.json".format(args.mwz_ver), 'w', encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--value_path", default="", type=str)
    parser.add_argument("--pretrain", default=r"t5-large", type=str)
    parser.add_argument("--mwz_ver", default=r"mwz2_1", type=str)
    args = parser.parse_args()
    stages = ["train", "dev", "test"]
    if args.value_path == "":
        for stage in stages:
            process(stage, args)
    else:
        process_value_gen(args)

