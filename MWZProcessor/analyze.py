import json


def func():
    with open("./data_processed/mwz2_1/train_raw.json", encoding="utf-8") as f:
        data = json.load(f)
    blank = 0
    for each in data:
        if len(each["turn_label"]) == 0:
            blank += 1
    print("Blank rate: {}".format(blank / len(data) * 100))


if __name__ == "__main__":
    func()
