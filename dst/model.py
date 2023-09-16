import json

import pytorch_lightning as pl
import torch
from transformers import get_linear_schedule_with_warmup


class DSTModel(pl.LightningModule):
    def __init__(self, args, model, tokenizer):
        super(DSTModel, self).__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def training_step(self, batch, batch_idx):
        output_prompt = self.model(input_ids=batch["input_id_prompt"], attention_mask=batch["attn_mask_prompt"],
                                   labels=batch["slot_encoded"])
        output_prompt_inverse = self.model(input_ids=batch["input_id_prompt_inverse"],
                                           attention_mask=batch["attn_mask_prompt_inverse"],
                                           labels=batch["value_encoded"])
        loss = output_prompt.loss + self.args.w * output_prompt_inverse.loss
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model.generate(input_ids=batch["input_id_prompt"], attention_mask=batch["attn_mask_prompt"])
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        ret = {
            "predict": output,
            "gt": batch["slot"]
        }
        return ret

    def validation_epoch_end(self, outputs):
        s = 0
        acc = 0
        for each in outputs:
            for idx, ea in enumerate(each["predict"]):
                if ea == each["gt"][idx]:
                    acc += 1
                s += 1
        self.log("single_acc", acc / s, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        output = self.model.generate(input_ids=batch["input_id_prompt"], attention_mask=batch["attn_mask_prompt"])
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        ret = {
            "generate": output,
            "value": batch["value"],
            "slot": batch["slot"],
            "flag": batch["flag"],
        }
        return ret

    def test_epoch_end(self, outputs):
        ans = []
        for each in outputs:
            for idx, ea in enumerate(each["value"]):
                ans.append({
                    "flag": each["flag"][idx],
                    "generate": each["generate"][idx],
                    "slot": each["slot"][idx],
                    "value": ea,
                })
        with open("./result/{}.json".format(self.args.result_name), 'w', encoding="utf-8") as f:
            json.dump(ans, f, ensure_ascii=False, indent=2)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000,
                                                    num_training_steps=int(self.args.n_epochs * self.args.num_train / (
                                                                self.args.train_batch_size * self.args.grad_batches)))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, "name": "LRMonitor"}
        return [optimizer], [scheduler]
