from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from torch.utils.data import DataLoader, Dataset


class KuCoinDataset(Dataset):
    def __init__(self, df: DataFrame, y: Series, window: int = 120, delay: int = 10):
        self.window = window
        self.delay = delay
        self.df = df[: -self.window].to_numpy()
        self.y = y[self.window + self.delay :].to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.df[idx : idx + self.window])
        y = torch.tensor(self.y[idx])
        return x, y


class BaseDataset:
    def __init__(self, day_of_month: int):
        self.day_of_month = day_of_month
        self.merged_df = self.data_preparation()

    def data_processing(self, path: Path, pair_name: str = "") -> DataFrame:
        col_for_drop = ["side", "trade_id", "trade_time"]
        df = pd.read_csv(path)
        df.trade_time = pd.to_datetime(df.trade_time, unit="ms")
        df.price = (df.price - df.price.min()) / (df.price.max() - df.price.min())
        df = df.set_index(df["trade_time"])
        df = df.drop(columns=col_for_drop)
        df = df.resample("S").mean()
        df.rename(
            columns={"price": f"price_{pair_name}", "size": f"size_{pair_name}"},
            inplace=True,
        )
        return df

    # Доделать нормально дату
    def data_preparation(self) -> DataFrame:
        p = Path.cwd()
        data_path = p.joinpath("data/")
        dataframes_dict = {}
        for file in data_path.rglob(f"*{self.day_of_month}.csv"):
            if file.is_file():
                pair_name = file.name[: file.name.find("USDT") + 4]
                dataframes_dict[pair_name] = self.data_processing(file, pair_name)
        merged_df = dataframes_dict["BTCUSDT"]  # Start with one DataFrame
        for key, df in dataframes_dict.items():
            if key != "BTCUSDT":  # Skip the first DataFrame
                merged_df = pd.merge(
                    merged_df, df, left_index=True, right_index=True, how="outer"
                )
        merged_df = merged_df.interpolate("linear")
        merged_df = merged_df.bfill()
        return merged_df

    def get_dataloader(self) -> DataLoader:
        dataset = KuCoinDataset(
            self.merged_df.drop(columns=["price_ETHUSDT", "size_ETHUSDT"]),
            self.merged_df[["price_ETHUSDT"]],
        )
        dataloader = DataLoader(dataset, batch_size=128, drop_last=True, shuffle=False)
        return dataloader

    def get_Xy(self) -> (np.array, np.array):
        dataloader = self.get_dataloader()
        X = list()
        y = list()
        for X_batch, y_batch in dataloader:
            for x_one in X_batch:
                X.append(x_one)
            for y_one in y_batch:
                y.append(y_one)
        X = np.array(X)
        y = np.array(y)
        return X, y
