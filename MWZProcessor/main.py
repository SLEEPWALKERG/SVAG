import argparse
import json
import random
from tqdm import tqdm


EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def data_processor(args, data):
    lst = []
    domain_counter = {}
    for sample in tqdm(data):
        dialogue_idx = sample["dialogue_idx"]
        domains = sample["domains"]
        for domain in domains:
            if domain not in EXPERIMENT_DOMAINS:
                continue
            if domain not in domain_counter:
                domain_counter[domain] = 0
            domain_counter[domain] += 1
        # Unseen domain setting
        if args["only_domain"] != "" and args["only_domain"] not in domains:
            continue
        if (args["except_domain"] != "" and args["stage"] == "test" and args["except_domain"] not in domains) or (
                args["except_domain"] != "" and args["stage"] != "test" and [args["except_domain"]] == domains):
            continue
        dial_history = []
        for turn in sample["dialogue"]:
            turn_idx = turn["turn_idx"]
            turn_domain = turn["domain"]
            if turn_domain not in EXPERIMENT_DOMAINS and args["remove_excluded_utterances"] == 1:
                continue
            turn_belief_state = turn["belief_state"]
            turn_label = turn["turn_label"]
            turn_system = turn["system_transcript"]
            turn_user = turn["transcript"]
            if args["stage"] != "test":
                if args["except_domain"] != "":
                    # Unseen domain setting in belief state
                    turn_belief_state = dict(
                        [(k, v) for k, v in turn_belief_state.items() if args["except_domain"] not in k])
                    # Unseen domain setting in turn label
                    turn_label = dict([(k, v) for k, v in turn_label.items() if args["except_domain"] not in k])
                elif args["only_domain"] != "":
                    # Unseen domain setting in belief state
                    turn_belief_state = dict([(k, v) for k, v in turn_belief_state.items() if args["only_domain"] in k])
                    # Unseen domain setting in turn label
                    turn_label = dict([(k, v) for k, v in turn_label.items() if args["only_domain"] in k])
            else:
                if args["except_domain"] != "":
                    # Unseen domain setting in belief state
                    turn_belief_state = dict(
                        [(k, v) for k, v in turn_belief_state.items() if args["except_domain"] in k])
                    # Unseen domain setting in turn label
                    turn_label = dict([(k, v) for k, v in turn_label.items() if args["except_domain"] in k])
                elif args["only_domain"] != "":
                    # Unseen domain setting in belief state
                    turn_belief_state = dict([(k, v) for k, v in turn_belief_state.items() if args["only_domain"] in k])
                    # Unseen domain setting in turn label
                    turn_label = dict([(k, v) for k, v in turn_label.items() if args["only_domain"] in k])
            lst.append({
                "domain": turn_domain,
                "belief_state": turn_belief_state,
                "turn_label": turn_label,
                "user": turn_user,
                "system": turn_system,
                "dialogue_idx": dialogue_idx,
                "turn_idx": turn_idx,
                "history": dial_history[:]
            })
            dial_history.append(turn_system)
            dial_history.append(turn_user)
    print("Number of {} samples: {}".format(args["stage"], len(lst)))
    print(domain_counter)
    return lst


def create_data(args):
    with open("./data/{}/{}_dials.json".format(args["mwz_ver"], args["stage"]), encoding="utf-8") as f:
        data = json.load(f)

    # determine training data ratio, default is 100%
    if args["stage"] == "train" and args["data_ratio"] != 100:
        random.Random(args["seed"]).shuffle(data)
        data_split = int(len(data) * 0.01 * args["data_ratio"])
        data_leave = data[data_split:]
        data = data[:data_split]
    lst = data_processor(args, data)
    with open("./data_processed/{}/{}_raw.json".format(args["mwz_ver"], args["stage"]), 'w', encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)
    if args["stage"] == "train" and args["data_ratio"] != 100:
        lst_leave = data_processor(args, data_leave)
        with open("./data_processed/{}/{}_leave_raw.json".format(args["mwz_ver"], args["stage"]), 'w', encoding="utf-8") as f:
            json.dump(lst_leave, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mwz_ver", default="mwz2_1", type=str)
    parser.add_argument("--only_domain", default="", type=str)
    parser.add_argument("--except_domain", default="", type=str)
    parser.add_argument("--stage", default="train", type=str)
    parser.add_argument("--data_ratio", default=1, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("remove_excluded_utterances", default=0, type=int)
    args = vars(parser.parse_args())
    create_data(args)
