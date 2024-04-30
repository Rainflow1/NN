from train import *
from argparse import *
from Trainer import Trainer
import torch
import os

if __name__ == "__main__":
    parser = ArgumentParser(description="Training")
    parser.add_argument("--dataset", default="./datasets/UCF_QRNF", help="preprocessed dataset path")
    parser.add_argument("--model", default="./checkpoint", help="model save path")

    parser.add_argument("--lr", type=float, default=1e-5, help="learn rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--crop-size", type=int, default=512)

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    for k, v in args.__dict__.items():
        print("Argument: {} -> {}".format(k, v))
    T = Trainer(args)
    T.train()
