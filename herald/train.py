import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from base_dataset import BaseDataset


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    params = OmegaConf.to_container(cfg["params"])
    data = BaseDataset(29)
    X, y = data.get_Xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, shuffle=False
    )
    model = linear_model.Ridge(**params)
    model.fit(X_train.reshape(-1, X_train.shape[1] * X_train.shape[2]), y_train)
    pred = model.predict(X_test.reshape(-1, X_test.shape[1] * X_test.shape[2]))
    print(f"mse:{mean_squared_error(y_test, pred)} r2_score: {r2_score(y_test, pred)}")


if __name__ == "__main__":
    train()
