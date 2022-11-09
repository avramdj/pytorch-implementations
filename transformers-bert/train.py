import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets import ImdbDataset
from models import BertClassifier

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

    parser = ArgumentParser()
    parser.add_argument(
        "dataset_path",
        help="Path to the imdb dataset. Download from kaggle `https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`",
    )
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument(
        "--num_workers", default=10, help="num threads to use for dataloaders"
    )
    args = parser.parse_args()
    assert args.epochs > 0, "more epochs pls"

    epochs = args.epochs
    batch_size = args.batch_size

    dataset = ImdbDataset(args.dataset_path)
    train_dataset, val_dataset = train_test_split(dataset, train_size=0.8)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=args.num_workers
    )
    trainer = pl.Trainer(max_epochs=epochs, accelerator=args.device)
    model = BertClassifier(n_classes=2, n_heads=12, n_layers=12)

    trainer.fit(model, train_loader, val_loader)
