import pytorch_lightning as pl
import torch
import json
from transformers import get_linear_schedule_with_warmup, WEIGHTS_NAME, CONFIG_NAME
import os
from copy import deepcopy


def metric_acc(predict, ground_truth):
    cnt = 0
    for idx, each in enumerate(predict):
        lst_predict = []
        for val in each.split(' | '):
            if val != ' ' and val != '':
                lst_predict.append(val)
        lst_predict.sort()
        lst_ground_truth = deepcopy(ground_truth[idx])
        lst_ground_truth.sort()
        if lst_predict == lst_ground_truth:
            cnt += 1
    return cnt


class T5ValueGenerator(pl.LightningModule):
    def __init__(self, args, model, tokenizer):
        super(T5ValueGenerator, self).__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.test_result = []

    def forward(self, batch):
        output = self.model(batch)
        return output

    def my_generate(self, input_ids, attn_mask):
        output = self.model.generate(input_ids=input_ids, attention_mask=attn_mask)
        predict = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return predict

    def training_step(self, batch, batch_idx):
        input_ids, attn_mask, labels = batch["input_ids"], batch["attn_mask"], batch["labels"]
        output = self.model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        return output.loss
    
    def training_epoch_end(self, outputs):
        loss = [each["loss"].item() for each in outputs]
        self.log("train_loss", sum(loss) / self.args.num_train, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        # input_ids, attn_mask, labels = batch["input_ids"], batch["attn_mask"], batch["labels"]
        # output = self.model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        batch_size, _ = batch["input_ids"].size()
        output = self.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attn_mask"])
        predict = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        correct_num = metric_acc(predict, batch["values"])
        return [correct_num, batch_size]

    def validation_epoch_end(self, outputs):
        correct = sum([each[0] for each in outputs])
        num = sum([each[1] for each in outputs])
        self.log("turn_acc", correct / num, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        output = self.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attn_mask"])
        predict = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        ret = {
            "flag": batch["flag"],
            "predict": predict,
            "values": batch["values"]
        }
        return ret

    def test_epoch_end(self, outputs):
        ans = []
        for each in outputs:
            for idx, ea in enumerate(each["values"]):
                ans.append({
                    "flag": each["flag"][idx],
                    "predict": each["predict"][idx],
                    "values": ea,
                })
        with open("./result/{}.json".format(self.args.result_name), 'w', encoding="utf-8") as f:
            json.dump(ans, f, ensure_ascii=False, indent=2)

    def save_model(self, path):
        output_model_file = os.path.join(path, WEIGHTS_NAME)
        output_config_file = os.path.join(path, CONFIG_NAME)
        torch.save(self.model.state_dict(), output_model_file)
        self.model.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(path)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.args.warm,
                                                    num_training_steps=int(
                                                        self.args.n_epochs * self.args.num_train / (
                                                                    self.args.train_batch_size * self.args.grad_batches)))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, "name": "LRMonitor"}
        return [optimizer], [scheduler]
