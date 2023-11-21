import tqdm
from typing import Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from deepimpression.common.metrics import update_confusion_matrix, metrics
from deepimpression.utils.util import save_model, deserialize
from deepimpression.utils.compute import get_device, data_parallel


def train(params: dict, model: Module, train_dl: DataLoader, test_dl: DataLoader):
    train_params = params['train_config']
    device = get_device()

    model = data_parallel(model)

    model.to(device)

    # Create optimizer and loss
    loss_cls = deserialize(train_params['loss']['class'])
    loss = loss_cls()
    opt_cls = deserialize(train_params['opt']['class'])
    opt = opt_cls(model.parameters(), lr=train_params['opt']['lr'])

    # Set training mode
    model.train()

    num_classes = train_params['num_classes']

    # Training loop
    for epoch in range(train_params["epochs"]):
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
        confusion_matrix = torch.zeros((num_classes, num_classes))
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

        full_message = f"Epoch {epoch+1}/{train_params['epochs']}: "
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
        save_model(model, params, train_params['modeldir'], weights_name=f"weights_{epoch+1:03d}.pth")
