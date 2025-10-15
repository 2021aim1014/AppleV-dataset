from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import pandas as pd
import os

def train_model(
    model_name,
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs=50,
    device='cuda'
):
    model = model.to(device)

    # count number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {num_params}")

    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation phase
        if epoch % 10 == 0:
            model.eval()
            val_loss = 0.0
            sample_count = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    sample_count += images.size(0)
            val_loss /= sample_count
            print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f"../saved_models/{model_name}_best.pth")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Testing phase
    model.eval()
    # load the best saved model
    print(f"Loading the best model for testing at path ../saved_models/{model_name}_best.pth")
    model.load_state_dict(torch.load(f"../saved_models/{model_name}_best.pth"))

    test_loss = 0.0
    with torch.no_grad():
        # train loss
        sample_count = 0
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            sample_count += images.size(0)

        train_loss /= sample_count
        print(f"Train Loss: {train_loss:.4f}")

        # val loss
        sample_count = 0
        val_loss = 0.0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            sample_count += images.size(0)
        val_loss /= sample_count
        print(f"Validation Loss: {val_loss:.4f}")

        # test loss
        sample_count = 0
        test_loss = 0.0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            sample_count += images.size(0)
    test_loss /= sample_count
    print(f"Test Loss: {test_loss:.4f}")

    # save number of parameters, train, val, test loss to a csv file
    import pandas as pd
    results = {
        "Model": model_name,
        "Num_Parameters": num_params,
        "Train_Loss": train_loss,
        "Validation_Loss": val_loss,
        "Test_Loss": test_loss
    }
    results_df = pd.DataFrame([results])
    if not os.path.exists("../results/csv/"):
        os.makedirs("../results/csv/")
    results_df.to_csv(f"../results/csv/{model_name}_results.csv", index=False)