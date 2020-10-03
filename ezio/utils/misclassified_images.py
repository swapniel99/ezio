import torch

def misclassified_images(model, device, test_loader):
    model.eval()
    result = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            prediction_ = pred.squeeze(-1)
            target_ = target.view_as(pred).squeeze(-1)
            for i in range(pred.size(0)):
              if prediction_[i]!=target_[i]:
                result.append([prediction_[i], target_[i], data[i]])
    return result