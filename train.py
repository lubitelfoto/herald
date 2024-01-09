import git
import hydra
import mlflow
from base_dataset import BaseDataset
from dvc.api import DVCFileSystem
from omegaconf import DictConfig, OmegaConf
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn import linear_model


mlflow.set_tracking_uri("http://128.0.1.1:8080")
mlflow.set_experiment("my_experiment")


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    with mlflow.start_run():
        params = OmegaConf.to_container(cfg["params"])

        mlflow.log_params(params)

        repo = git.Repo(search_parent_directories=True)
        git_commit_id = repo.head.object.hexsha
        mlflow.log_param("git_commit_id", git_commit_id)

        fs = DVCFileSystem()
        fs.repo.pull()

        data = BaseDataset(29)
        X_train, y_train = data.get_Xy()
        model = linear_model.Ridge(**params)
        model.fit(X_train.reshape(-1, X_train.shape[1] * X_train.shape[2]), y_train)
        initial_type = [
            ("float_input", FloatTensorType([None, X_train.shape[1] * X_train.shape[2]]))
        ]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open("models/model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        f.close()

    # Пушить модель через апи пока не получилось
    # fs.repo.push()


if __name__ == "__main__":
    train()
