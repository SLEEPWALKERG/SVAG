import argparse
import json
import re
import numpy as np
from copy import deepcopy
import random
random.seed(42)


def fix_time_label(value):
    x = re.search(r"\d\d\D\d\d", value)
    if x is not None:
        x = x.group()
        biaodian = re.search(r"\D", x).group()
        if biaodian != ":":
            return value.replace(biaodian, ':')
    return value


def analyze(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # random.shuffle(data)
    # data = data[:15595]
    j_acc = 0
    j = 0
    turn_num = 0
    acc = 0
    blank = 0
    lst = []
    for each in data:
        turn_num += 1
        lst_predict = []
        for pv in each["predict"].split(' | '):
            lst_predict.append(fix_time_label(pv))
        lst_p = []
        for x in lst_predict:
            if x == ' ' or x== '':
                continue
            else:
                lst_p.append(x)
        if len(lst_p) == 0:
            blank += 1
        lst_g = deepcopy(each["values"])
        lst_p.sort()
        lst_g.sort()
        if lst_p == lst_g:
            j += 1
        # lst_predict = set(each["predict"].split(' | ')) - {' ', ''}
        lst_predict = set(lst_predict) - {' ', ''}
        # lst_predict = set()
        lst_ground_truth = set(each["values"])
        if lst_predict == lst_ground_truth: #and len(lst_p) == len(each["values"]):
            j_acc += 1
        else:
            lst.append({
                "flag": each["flag"],
                "predict": list(lst_predict),
                "ground_truth": list(lst_ground_truth)
            })
        if len(lst_ground_truth - lst_predict) == 0:
            acc += 1
    with open("./result/err.json", 'w', encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)
    print(j / turn_num)
    print(blank / turn_num)
    print("完全匹配的正确率为：{:.2f} %".format(j_acc / turn_num * 100))
    print("仅正确值包含预测值的正确率为：{:.2f} %".format(j_acc / turn_num * 100))


def watch_err(path):
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    with open("../MWZProcessor/data_processed/mwz2_1/test_raw.json", encoding="utf-8") as f:
        data = json.load(f)
    dic = {}
    for each in data:
        dic[each["dialogue_idx"] + '-' + str(each["turn_idx"])] = each
    lst = []
    for r in result:
        raw = dic[r["flag"]]
        lst.append({
            "flag": r["flag"],
            "history": raw["history"],
            "system": raw["system"],
            "user": raw["user"],
            "predict": r["predict"],
            "ground_truth": r["ground_truth"]
        })
    with open("./result/watch_err.json", 'w', encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)


def err_turn():
    with open("./result/err.json", encoding="utf-8") as f:
        data = json.load(f)
    lst = []
    for each in data:
        dialogue_idx, turn_idx = each["flag"].split('-')
        lst.append(int(turn_idx))
    print(len(lst))
    print("average turn idx: {}".format(np.mean(lst)))
    print("median turn idx: {}".format(np.median(lst)))
    counts = np.bincount(lst)
    print("zhongshu: {}".format(np.argmax(counts)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default="./result/result.json", type=str)
    args = parser.parse_args()
    analyze(args.result_path)
    watch_err("./result/err.json")
    err_turn()
