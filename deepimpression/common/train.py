from torch_data import CustomDataset
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from torchsummary import summary
from typing import Dict, Optional
import pickle
import tqdm
import numpy as np
import argparse

from deepimpression.utils import load_model, save_model, deserialize_class
from deepimpression.metrics import update_confusion_matrix, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", help="model directory to use", type=str, required=True)
    args = parser.parse_args()

    # Load parameters
    model, params = load_model(args.modeldir)

    # Data Preparation
    ## Load data from numpy files
    X_train = np.load(params['train_input']['x'])['arr_0']
    y_train = np.load(params['train_input']['y'])['arr_0']
    X_test = np.load(params['eval_inupt']['x'])['arr_0']
    y_test = np.load(params['eval_input']['y'])['arr_0']

    ## Compute some statistics about the data
    ic(y_train.shape)
    num_classes = len(np.unique(y_train))
    ### Get class distribution
    class_freq = np.bincount(y_train.astype(np.int32))
    ### Get class weights as array
    class_weights = np.divide(np.ones(num_classes), class_freq)
    class_weights = class_weights / np.min(class_weights)

    ic(class_weights)

    ## Swap last two axes so channels are first
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)

    ## binarize labels 
    y_train = OneHotEncoder(max_categories=3, sparse_output=False).fit_transform(y_train.reshape(-1, 1))
    y_test = OneHotEncoder(max_categories=3, sparse_output=False).fit_transform(y_test.reshape(-1, 1))

    batch_size = 32

    train_ds = CustomDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = CustomDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    loss_cls = deserialize_class(params['loss']['class'])
    loss = loss_cls(weight=torch.Tensor(class_weights).to(device))
    opt_cls = deserialize_class(params['opt']['class'])
    opt = opt_cls(model.parameters(), lr=params['opt']['learning_rate'])

    # Set training mode
    model.train()

    # Training loop
    for epoch in range(params['runconfig']["epochs"]):
        # Train
        all_loss = 0.
        # confusion matrix
        confusion_matrix = torch.zeros((num_classes, num_classes))

        for X, Y in tqdm.tqdm(train_dl):
            # Move to device
            X, Y = X.to(device), Y.to(device)

            # Forward pass
            outs = model(X)
            loss_val = loss(outs, Y.float())
            all_loss += loss_val.item()

            # Fill confusion matrix
            predicted_classes = outs.argmax(1)
            true_classes = Y.argmax(1)
            update_confusion_matrix(confusion_matrix, true_classes, predicted_classes)

            # Backwards pass and optimization
            opt.zero_grad()
            loss_val.backward()
            opt.step()

        train_loss = all_loss/len(train_dl)
        train_metrics = metrics(confusion_matrix)

        # Compute loss and accuracy over validation set
        all_loss = 0.
        confusion_matrix = torch.zeros((params['num_classes'], params['num_classes']))
        for X, Y in test_dl:
            # Move to device
            X, Y = X.to(device), Y.to(device)

            # Forward pass and loss computation
            outs = model(X)
            loss_val = loss(outs, Y.float())
            all_loss += loss_val.item()

            # Fill confusion matrix
            predicted_classes = outs.argmax(1)
            true_classes = Y.argmax(1)
            update_confusion_matrix(confusion_matrix, true_classes, predicted_classes)

        test_loss = all_loss/len(test_dl)
        test_metrics = metrics(confusion_matrix)

        full_message = f"Epoch {epoch+1}/{params['epochs']}: "
        full_message += "Training: "
        full_message += "Loss: {:.4f}, ".format(train_loss)
        for key, val in train_metrics.items():
            full_message += "{}: {:.4f}, ".format(key, val)
        full_message += "Test: "
        full_message += "Loss: {:.4f}, ".format(test_loss)
        for key, val in test_metrics.items():
            full_message += "{}: {:.4f}, ".format(key, val)
        print(full_message)

        # Save model to disk
        save_model(model, params, args.modeldir, weights_name=f"weights_{epoch+1:03d}.pth")


if __name__ == "__main__":
    main()
