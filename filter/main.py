from transformers import RobertaForSequenceClassification
import pytorch_lightning as pl
import argparse
from roberta_nli import RobertaNLI
from dataset import NLIDataSet
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train(args):
    pl.seed_everything(42)
    pretrain_model = RobertaForSequenceClassification.from_pretrained(args.pretrain, num_labels=args.num_labels)
    model = RobertaNLI(args, pretrain_model)
    logger = TensorBoardLogger("logs", name=args.name)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(dirpath="./save/{}".format(args.name), save_top_k=5, monitor="f05",
                                          mode="max")
    data = NLIDataSet(args)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
        gpus=1,
        check_val_every_n_epoch=1,
        max_epochs=args.n_epochs,
        default_root_dir=args.save_dir,
        accumulate_grad_batches=args.grad_batches,
    )
    print("Start training ...")
    trainer.fit(model, data)
    # print("Start testing ...")
    # trainer.test(model, data)
    with open("./best_model_{}.txt".format(args.name), 'w', encoding="utf-8") as f:
        f.write(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--dev_batch_size", type=int, default=16)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--pretrain", type=str, default=r"roberta-base")
    parser.add_argument("--mwz_ver", type=str, default=r"mwz2_1")
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--grad_batches", type=int, default=8)
    parser.add_argument("--name", type=str, default="nd")
    parser.add_argument("--monitor", type=str, default="precision")
    parser.add_argument("--num_train", default=10000, type=int)
    args = parser.parse_args()
    train(args)
