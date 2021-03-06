from ezio.train.trainer import train
from ezio.eval.evaluator import test


def fit(model, train_loader, valid_loader, loss_function, device, optimizer, epochs, scheduler=None):
    train_acc = []
    train_losses = []
    test_acc = []
    test_losses = []
    learning_rate = []

    for epoch in range(epochs):
        print("EPOCH:", epoch + 1)
        ta, tl, lr = train(model, device, train_loader, optimizer, loss_function, scheduler=scheduler)
        train_acc += ta
        train_losses += tl
        learning_rate += lr

        test_loss, test_acc_, test_losses_ = test(model, device, valid_loader, loss_function)
        test_acc += test_acc_
        test_losses += test_losses_

    if scheduler:
        return train_acc, train_losses, test_acc, test_losses, learning_rate
    else:
        return train_acc, train_losses, test_acc, test_losses