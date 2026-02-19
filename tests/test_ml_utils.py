import numpy as np
from ml_utils import get_model, train_and_evaluate

def test_get_model():
    # ensure that valid names return objects
    names = ["Logistic Regression", "Decision Tree", "Random Forest", "SVC"]
    for name in names:
        model = get_model(name)
        assert model is not None


def test_train_and_evaluate():
    # generate a simple dataset
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    model = get_model("Decision Tree", max_depth=2)
    model, metrics = train_and_evaluate(model, X, X, y, y)
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0
