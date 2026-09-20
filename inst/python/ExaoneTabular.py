from pathlib import Path

import numpy as np
import torch
from exaonetabular import EXAONETabularClassifier


def fit_model(features, targets, device, seed):
    model = EXAONETabularClassifier.from_pretrained(device=device, seed=seed)
    model.fit(features, targets)
    return model


def predict_proba(model, features):
    probabilities = model.predict_proba(features)
    positive_class = np.flatnonzero(model.classes_ == 1)[0]
    return probabilities[:, positive_class]


def get_selected_feature_indices(model):
    selected_columns = model.selected_feature_indices_
    if selected_columns is None:
        selected_columns = np.arange(model.n_features_in_)
    return selected_columns


def save_model(model, path):
    # A state_dict alone omits EXAONE's context, preprocessing and selected columns.
    torch.save(model, Path(path) / "ExaoneTabularModel.pt")


def load_model(path):
    return torch.load(Path(path) / "ExaoneTabularModel.pt", weights_only=False)
