from ezio.train.trainer import train
from ezio.eval.evaluator import test



def fit(model, train_loader, valid_loader, loss_function, device, optimizer, epochs, scheduler=None):
    for epoch in range(epochs):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optimizer, loss_function)
        test_loss = test(model, device, valid_loader, loss_function)
        if scheduler:
            scheduler.step(test_loss)