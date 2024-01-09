import numpy as np
import onnxruntime as rt
from base_dataset import BaseDataset
from dvc.api import DVCFileSystem
from sklearn.metrics import mean_squared_error, r2_score


def infer():
    fs = DVCFileSystem()
    fs.repo.pull()
    test_data = BaseDataset(30)
    X, y = test_data.get_Xy()
    X = X.astype(np.float32)
    sess = rt.InferenceSession("models/model.onnx")
    input_name = sess.get_inputs()[0].name
    input_data = {input_name: X.reshape(-1, X.shape[1] * X.shape[2])}
    pred = sess.run(None, input_data)[0]
    np.savetxt("infer_output/output.csv", pred, delimiter=",", fmt="%f")
    print(f"infer mse:{mean_squared_error(y, pred)} r2_score: {r2_score(y, pred)}")


if __name__ == "__main__":
    infer()
