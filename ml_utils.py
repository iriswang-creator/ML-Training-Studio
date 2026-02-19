from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def get_model(name: str, **kwargs):
    """Return a scikit-learn model instance based on name and hyperparameters"""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, **kwargs)
    elif name == "Decision Tree":
        return DecisionTreeClassifier(**kwargs)
    elif name == "Random Forest":
        return RandomForestClassifier(**kwargs)
    elif name == "SVC":
        return SVC(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return model, {"accuracy": acc, "classification_report": report, "confusion_matrix": cm}


def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
