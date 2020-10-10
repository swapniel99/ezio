import torch
from tqdm import tqdm

def train(model, device, train_loader, optimizer, criterion, l1_loss=False, lambda_l1=None, scheduler=None):
    train_losses = []
    train_acc = []
    learning_rate = []

    model.train()
    pbar = tqdm(train_loader)

    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()

        y_pred = model(data)
        loss = criterion(y_pred, target)
        if l1_loss:
            l1 = 0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
            loss = loss + lambda_l1 * l1
        train_losses.append(loss)

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        _, preds = torch.max(y_pred, 1)
        correct += torch.sum(preds == target.data)
        processed += len(data)

        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={(100 * correct // processed):0.2f} Learning Rate={optimizer.state_dict()["param_groups"][0]["lr"]}')
        train_acc.append(100 * correct // processed)
        learning_rate.append(optimizer.state_dict()["param_groups"][0]["lr"])
    return train_acc, train_losses, learning_rate