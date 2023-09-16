import argparse
import json
import random
from copy import deepcopy
random.seed(42)


def filter_analyze(args, data):
    with open(args.score_file_path, encoding="utf-8") as f:
        score = json.load(f)
    cnt = 0
    num_blank = 0
    for i, each in enumerate(data):
        if score[i] > args.threshold:
            cnt += 1
            if len(each["lst_predict"]) == 0:
                num_blank += 1
    print("----------------------------roberta filter-------------------------------")
    print("The threshold is {}".format(args.threshold))
    print("Count of pseudo labels: {}".format(cnt))
    print("Blank rate: {:.2f} %".format(num_blank / cnt * 100))
    print("-------------------------------------------------------------------------")
    return cnt


def random_analyze(data, cnt):
    tmp = deepcopy(data)
    random.shuffle(tmp)
    tmp = tmp[:cnt]
    acc = 0
    num_blank = 0
    for each in tmp:
        if len(each["lst_predict"]) == 0:
            num_blank += 1
    print("----------------------------random filter-------------------------------")
    print("Acc : {:.2f} %".format(acc / cnt * 100))
    print("Count of pseudo labels: {}".format(cnt))
    print("Blank rate: {:.2f} %".format(num_blank / cnt * 100))
    print("-------------------------------------------------------------------------")


def main(args):
    with open(args.gen_file_path, encoding="utf-8") as f:
        data = json.load(f)
    for each in data:
        lst_predict = []
        for val in each["predict"].split(' | '):
            if val != ' ' and val != '':
                lst_predict.append(val)
        each["lst_predict"] = lst_predict
    cnt = filter_analyze(args, data)
    random_analyze(data, cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_file_path", type=str, default="../filter/generate/level_1.json")
    parser.add_argument("--gen_file_path", type=str, default="./generate/level_1.json")
    parser.add_argument("--threshold", type=float, default=0.98)
    args = parser.parse_args()
    main(args)
