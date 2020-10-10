import math
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

start_lr = 1e-7
lr_find_epochs = 5
end_lr = 1

INCREASE_MODE = 'exponential'

def lr_range(model, train_loader, device):
    all_lrs = []
    all_lr_loss = []
    all_lr_acc = []

    iter = 0
    smoothing = 0.05

    if INCREASE_MODE == 'exponential':
      # exponential increase
      optimizer = optim.SGD(model.parameters(), start_lr)
      lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / (lr_find_epochs * len(train_loader)))

    else:
      # linear increase
      optimizer = optim.SGD(model.parameters(), 1.)
      total_iterations = (len(train_loader)) * lr_find_epochs
      slope = (end_lr - start_lr) / (total_iterations)
      lr_lambda = lambda x: ((slope * x) + start_lr)


    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(lr_find_epochs):
        pbar = tqdm(train_loader)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            y_pred = model(data)

            loss = criterion(y_pred, target)

            loss.backward()
            optimizer.step()

            # update lr
            scheduler.step()
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            all_lrs.append(lr_step)

            # loss
            if iter==0:
              all_lr_loss.append(loss)
            else:
              loss = smoothing  * loss + (1 - smoothing) * all_lr_loss[-1]
              all_lr_loss.append(loss)

            # accuracy
            _, preds = torch.max(y_pred, 1)
            correct = torch.sum(preds == target.data)
            processed = len(data)
            accuracy = 100 * correct // processed

            all_lr_acc.append(accuracy)

            iter+=1

    return all_lrs, all_lr_loss, all_lr_acc