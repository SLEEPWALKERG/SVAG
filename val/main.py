from transformers import T5ForConditionalGeneration, T5Tokenizer
import pytorch_lightning as pl
import argparse
from model import T5ValueGenerator
from dataset import MWZDataSet
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train(args):
    pl.seed_everything(42)
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain)
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain)
    logger = TensorBoardLogger("logs", name=args.name)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(dirpath="./save/{}".format(args.name), save_top_k=2, monitor="turn_acc",
                                          mode="max")
    data = MWZDataSet(args)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
        gpus=1,
        # val_check_interval=5000,
        max_epochs=args.n_epochs,
        # max_steps=20000,
        default_root_dir=args.save_dir,
        accumulate_grad_batches=args.grad_batches  
    )
    if args.is_train == 1:
        model = T5ValueGenerator(args, pretrain_model, tokenizer)
        print("Start training ...")
        trainer.fit(model, data)
        model_path = checkpoint_callback.best_model_path
    else:
        model_path = args.model_path
    model = T5ValueGenerator.load_from_checkpoint(model_path, args=args, model=pretrain_model, tokenizer=tokenizer)
    print("Start testing ...")
    trainer.test(model, data)
    print(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--dev_batch_size", type=int, default=32)
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--pretrain", type=str, default=r"t5-large")
    parser.add_argument("--mwz_ver", type=str, default=r"mwz2_1")
    parser.add_argument("--grad_batches", type=int, default=4)
    parser.add_argument("--is_train", default=1, type=int)
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--result_name", default="result", type=str)
    parser.add_argument("--num_train", default=578, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--warm", default=100, type=int)
    parser.add_argument("--level_1", default="", type=str)
    parser.add_argument("--level_2", default="", type=str)
    args = parser.parse_args()
    train(args)
