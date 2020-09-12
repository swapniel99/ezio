import torch
from tqdm import tqdm



def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    lambda_l1 = 0.005
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)
        # l1 = 0
        # for p in model.parameters():
        #   l1 = l1 + p.abs().sum()
        # loss = loss + lambda_l1 * l1

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar_str = f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}'
        pbar.set_description(desc= pbar_str)


def test(model, device, test_loader, criterion, classes, test_losses, test_accs, misclassified_imgs, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss +=criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            is_correct = pred.eq(target.view_as(pred))
            correct += is_correct.sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    test_acc = 100. * correct / len(test_loader.dataset)
    test_accs.append(test_acc)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))

    return test_loss, test_accs, misclassified_imgs