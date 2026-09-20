import numpy as np
import torch
from exaonetabular import EXAONETabularClassifier


def fit_exaone_tabular(features, targets, device, seed):
    model = EXAONETabularClassifier.from_pretrained(device=device, seed=seed)
    model.fit(features, targets)
    return model


def predict_exaone_tabular(model, features):
    probabilities = model.predict_proba(features)
    positive_class = np.flatnonzero(model.classes_ == 1)[0]
    return probabilities[:, positive_class]


def save_exaone_tabular(model, path):
    # A state_dict alone omits EXAONE's context, preprocessing and selected columns.
    torch.save(model, path)


def load_exaone_tabular(path):
    return torch.load(path, weights_only=False)
