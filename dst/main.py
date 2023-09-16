from transformers import T5ForConditionalGeneration, T5Tokenizer
import pytorch_lightning as pl
import argparse
from model import DSTModel
from dataset import DSTDataset
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train(args):
    pl.seed_everything(42)
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain)
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain)
    logger = TensorBoardLogger("logs", name=args.name)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(dirpath="./save/{}".format(args.name), save_top_k=3, monitor="single_acc",
                                          mode="max")
    data = DSTDataset(args)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
        gpus=1,
        check_val_every_n_epoch=1,
        max_epochs=args.n_epochs,
        default_root_dir=args.save_dir,
        accumulate_grad_batches=args.grad_batches,
    )
    if args.is_train == 1:
        model = DSTModel(args, pretrain_model, tokenizer)
        print("Start training ...")
        trainer.fit(model, data)
        model_path = checkpoint_callback.best_model_path
    else:
        model_path = args.model_path
    model = DSTModel.load_from_checkpoint(
        model_path,
        args=args, model=pretrain_model, tokenizer=tokenizer
    )
    print("Start testing ...")
    trainer.test(model, data)
    print(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", default=r"t5-large", type=str)
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--dev_batch_size", default=16, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int)
    parser.add_argument("--n_workers", default=4, type=int)
    parser.add_argument("--n_epochs", default=20, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--save_dir", default=r"./save", type=str)
    parser.add_argument("--mwz_ver", default="mwz2_1", type=str)
    parser.add_argument("--w", default=0.1, type=float)
    parser.add_argument("--name", default="dst", type=str)
    parser.add_argument("--num_train", default=0, type=int)
    parser.add_argument("--grad_batches", default=4, type=int)
    parser.add_argument("--is_train", default=1, type=int)
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--result_name", default="result", type=str)
    parser.add_argument("--test_name", default="test", type=str)
    args = parser.parse_args()
    train(args)
