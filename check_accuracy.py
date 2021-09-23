import torch
import torch.nn.functional as F
def check_accuracy(loader, model, device):
    dtype = torch.float32
    # if loader.dataset.train:
    #     print('Checking accuracy on validation set')
    # else:
    #     print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    loss_accumulate = 0.0
    model.eval()   # set model to evaluation mode
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _,preds = scores.max(1)
            num_correct += (preds==y).sum()
            num_samples += preds.size(0)

            loss = F.cross_entropy(scores, y)
            loss_accumulate += loss.item()

        acc = float(num_correct) / num_samples
        loss = loss_accumulate / num_samples
        # print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 *acc ))
        return acc, loss
