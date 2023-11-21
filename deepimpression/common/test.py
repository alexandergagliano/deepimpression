from deepimpression.t2model_v1.model import T2Model_AG
from deepimpression.data import CustomDataset
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


def save_weights(model: nn.Module, filename: str) -> None:
    # Saves weights to the given filename
    with open(filename, 'wb') as f:
        pickle.dump(model.state_dict(), f)


def metrics(confusion_matrix: torch.Tensor) -> dict[str, float]:
    # Compute metrics from the confusion matrix. format is: [true class, predicted class]
    if confusion_matrix.ndim != 2 or confusion_matrix.size(0) != confusion_matrix.size(1):
        raise ValueError("Confusion matrix must be a square matrix.")
    
    num_classes = confusion_matrix.size(0)
    result: Dict[str, float] = {}

    # Plain Accuracy
    total = torch.sum(confusion_matrix).float()
    result['acc'] = (torch.trace(confusion_matrix) / total).item() if total > 0 else 0.0

    # F1 scores
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = torch.sum(confusion_matrix[:, i]) - tp
        fn = torch.sum(confusion_matrix[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result[f"f1({i})"] = f1.item()
    
    return result


def update_confusion_matrix(
        confusion_matrix: torch.Tensor,
        true_classes: torch.Tensor,
        predicted_classes: torch.Tensor,
        class_weights: Optional[torch.Tensor]=None) -> None:
    
    # Update the confusion matrix with the given true and predicted classes
    for t_i, p_i in zip(true_classes, predicted_classes):
        confusion_matrix[t_i, p_i] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", help="Checkpoint to use", type=str, required=True)
    args = parser.parse_args()

    # Load data from numpy files
    X_train = np.load('X_train_ZTF_Sim_FullSliced_Padded_0pt2GP_hostPhotTrue_30Cut.npz')['arr_0']
    y_train = np.load('y_train_ZTF_Sim_FullSliced_Padded_0pt2GP_hostPhotTrue_30Cut.npz')['arr_0']
    X_test = np.load('X_test_ZTF_Sim_FullSliced_Padded_0pt2GP_hostPhotTrue_30Cut.npz')['arr_0']
    y_test = np.load('y_test_ZTF_Sim_FullSliced_Padded_0pt2GP_hostPhotTrue_30Cut.npz')['arr_0']

    # Compute some statistics about the data
    ic(y_train.shape)
    num_classes = len(np.unique(y_train))
    # Get class distribution
    class_freq = np.bincount(y_train.astype(np.int32))
    # Get class weights as array
    class_weights = np.divide(np.ones(num_classes), class_freq)
    class_weights = class_weights / np.min(class_weights)

    ic(class_weights)

    # Swap last two axes so channels are first
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)

    #binarize labels 
    y_train = OneHotEncoder(max_categories=3, sparse_output=False).fit_transform(y_train.reshape(-1, 1))
    y_test = OneHotEncoder(max_categories=3, sparse_output=False).fit_transform(y_test.reshape(-1, 1))

    batch_size = 32

    train_ds = CustomDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = CustomDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model

    params = {}
    # using the optimally-determined parameters here
    params['num_classes'] = num_classes
    params['batch_size'] = batch_size
    params['epochs'] = 200
    #params['epochs'] = 10
    #params['epochs'] = 1
    params['embed_dim'] = 512
    params['ff_dim'] = params['embed_dim']*6
    params['num_layers'] = 8
    params['num_heads'] = 4
    params['droprate'] = 0.28

    #params = {}
    ## using the optimally-determined parameters here
    #params['num_classes'] = num_classes
    #params['batch_size'] = batch_size
    #params['epochs'] = 200
    ##params['epochs'] = 10
    ##params['epochs'] = 1
    #params['embed_dim'] = 32
    #params['ff_dim'] = params['embed_dim']*4
    #params['num_layers'] = 4
    #params['num_heads'] = 2
    #params['droprate'] = 0.28

    # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
    params['num_filters'] = params['embed_dim']
    params['passbands'] = 'XY'

    (_, num_features, timesteps,) = X_train.shape 
    input_shape = (params['batch_size'], num_features, timesteps)
    params['input_shape'] = input_shape

    # Create the TF model object
    model = T2Model_AG(
        input_dim=params['input_shape'],
        embed_dim=params['embed_dim'],
        num_heads=params['num_heads'],
        ff_dim=params['ff_dim'],
        num_filters=params['num_filters'],
        num_classes=params['num_classes'],
        num_layers=params['num_layers'],
        droprate=params['droprate'],
    )

    # Set eval mode
    model.eval()

    # Load weights
    with open(args.model_file, 'rb') as f:
        old_state_dict = pickle.loads(f.read())
        # Remove module prefix since we aren't using DataParallel here.
        new_state_dict = {k[len("module."):]: v for k, v in old_state_dict.items() if k.startswith("module.")}
        model.load_state_dict(new_state_dict)

    model.to(device)

    loss = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))

    # Compute loss and accuracy over validation set
    all_loss = 0.
    confusion_matrix = torch.zeros((params['num_classes'], params['num_classes']))
    for X, Y in tqdm.tqdm(test_dl):
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

    full_message = "Test: "
    full_message += "Loss: {:.4f}, ".format(test_loss)
    for key, val in test_metrics.items():
        full_message += "{}: {:.4f}, ".format(key, val)
    print(full_message)
    # Normalize confusion matrix with respect to first index
    confusion_matrix_norm1 = torch.zeros((params['num_classes'], params['num_classes']))
    for i in range(params['num_classes']):
        confusion_matrix_norm1[i] = confusion_matrix[i] / confusion_matrix[i].sum()
    confusion_matrix_norm2 = torch.zeros((params['num_classes'], params['num_classes']))
    # Normalize confusion matrix with respect to second index
    for i in range(params['num_classes']):
        confusion_matrix_norm2[:, i] = confusion_matrix[:, i] / confusion_matrix[:, i].sum()
    print("confusion_matrix:")
    print(confusion_matrix)
    print("confusion_matrix_norm1:")
    print(confusion_matrix_norm1)
    print("confusion_matrix_norm2:")
    print(confusion_matrix_norm2)



if __name__ == "__main__":
    main()