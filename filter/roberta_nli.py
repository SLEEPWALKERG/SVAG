import pytorch_lightning as pl
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics


# NSP-NLI
class RobertaNLI(pl.LightningModule):
    def __init__(self, args, pretrain):
        super(RobertaNLI, self).__init__()
        self.args = args
        self.model = pretrain

    def forward(self, batch, batch_idx):
        input_ids, attn_mask, type_ids, labels = batch["input_ids"], batch["attn_mask"], batch["type_ids"], batch[
            "labels"]
        output = self.model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        return output

    def my_generate(self, batch, batch_idx):
        with torch.no_grad():
            out = self(batch, batch_idx)
        prob = F.softmax(out.logits, dim=-1)
        acc = prob[:, 2]
        return acc

    def training_step(self, batch, batch_idx):
        out = self(batch, batch_idx)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self(batch, batch_idx)
        out = torch.argmax(out.logits, dim=-1)
        ret = {
            "predict": out,
            "ground_truth": batch["labels"],
        }
        return ret

    def validation_epoch_end(self, outputs):
        pred = torch.concat([each["predict"] for each in outputs], dim=-1)
        gt = torch.concat([each["ground_truth"] for each in outputs], dim=-1)
        f_05 = metrics.fbeta_score(pred, gt, beta=0.5, average="none", num_classes=3)[2].item()
        f_1 = metrics.fbeta_score(pred, gt, beta=1.0, average="none", num_classes=3)[2].item()
        f_2 = metrics.fbeta_score(pred, gt, beta=2.0, average="none", num_classes=3)[2].item()
        precision = metrics.precision(pred, gt, average="none", num_classes=3)[2].item()
        acc = metrics.accuracy(pred, gt).item()
        self.log("f05", f_05, on_epoch=True, logger=True)
        self.log("f1", f_1, on_epoch=True, logger=True)
        self.log("f2", f_2, on_epoch=True, logger=True)
        self.log("precision", precision, on_epoch=True, logger=True)
        self.log("acc", acc, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        out = self(batch, batch_idx)
        out = torch.argmax(out.logits, -1)
        ret = {
            "predict": out,
            "ground_truth": batch["labels"],
        }
        return ret

    def test_epoch_end(self, outputs):
        s = 0
        acc = 0
        for each in outputs:
            pre = torch.eq(each["predict"], 2)
            real = torch.sum(torch.eq(pre * each["ground_truth"], 2))
            pre_count = torch.sum(pre)
            s += pre_count.item()
            acc += real.item()
        with open("./result.txt", 'w', encoding="utf-8") as f:
            f.write("总共预测了正面例子有: {}\n".format(s))
            f.write("预测是正确的例子里面真的正确的比例: {}".format(acc / s))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000,
                                                    num_training_steps=int(
                                                        self.args.n_epochs * self.args.num_train / (
                                                                    self.args.train_batch_size * self.args.grad_batches)))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, "name": "LRMonitor"}
        return [optimizer], [scheduler]
