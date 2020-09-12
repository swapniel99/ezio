import torch


def test(model, device, test_loader, criterion):
    test_losses = []
    test_acc = []
    model.eval()
    test_loss = 0
    correct = 0
    # criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)  # sum up batch loss
            _, preds = torch.max(output, 1)
            test_loss += loss.item()
            correct += torch.sum(preds == target.data)
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

    return test_loss, test_acc, test_losses
