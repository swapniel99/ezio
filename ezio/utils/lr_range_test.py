import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import torch.optim as optim


def lr_range(input_model, device, total_epochs, train_loader, criterion, lrmax, lrmin):
    # Step size to increase the learning rate over every epoch
    step_size = (lrmax - lrmin)/total_epochs
    learning_rates = list()
    training_accuracies = list()
    # Initial running rate
    learning_rate = lrmin
    for current_epoch in range(total_epochs):
        print('Learning rate:',learning_rate)
        model = copy.deepcopy(input_model)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        model.train()
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()
            y_pred = model(data)
            loss = criterion(y_pred, target)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(y_pred, 1)  # taking the highest value of prediction.
            correct += torch.sum(
                preds == target.data)  # calculating te accuracy by taking the sum of all the correct predictions in a batch.
            processed += len(data)
            pbar.set_description(
                desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={(100 * correct // processed):0.2f}')
        training_accuracies.append(100*correct//processed)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        learning_rate += step_size


    # Plot the graph between accuracy vs learning rate
    plt.plot(learning_rates, training_accuracies)
    plt.ylabel('train Accuracy')
    plt.xlabel("Learning rate")
    plt.title("Lr v/s accuracy")
    plt.show()
