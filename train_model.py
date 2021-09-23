import torch
import torch.nn.functional as F
from check_accuracy import check_accuracy
import copy
from tensorboardX import SummaryWriter
def train_model(model, optimizer, loader_train, loader_val, device, tb_saved_path, epochs=1, scheduler=None):
    writer = SummaryWriter(tb_saved_path)
    dtype = torch.float32
    best_model_wts = None
    best_acc = 0.0
    model = model.to(device=device) # move the model parameters to CPU/GPU
    for epoch in range(epochs):
        if scheduler:
            scheduler.step()
        for t,(x,y) in enumerate(loader_train):
            model.train()   # set model to training mode
            x = x.to(device, dtype=dtype)
            y = y.to(device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_acc, val_loss = check_accuracy(loader_val, model, device)
        train_acc, train_loss = check_accuracy(loader_train, model, device)
        print('Epoch:%d\ttrain_loss=%.4f,\tval_loss=%.4f\n\ttrain_acc=%.4f,\tval_acc=%.4f'
              % (epoch, train_loss, val_loss, train_acc, val_acc))
        # writer.add_scalar('train_acc', train_acc, global_step=epoch)
        # writer.add_scalar('val_acc', val_acc, global_step=epoch)
        # writer.add_scalar('train_loss', train_loss, global_step=epoch)
        # writer.add_scalar('val_loss', val_loss, global_step=epoch)
        writer.add_scalars("loss", {'train_loss': train_loss, 'val_loss': val_loss}, global_step=epoch)
        writer.add_scalars("acc", {'train_acc': train_acc, 'val_acc': val_acc}, global_step=epoch)

        if val_acc > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = val_acc
    print('best_acc:', best_acc)
    model.load_state_dict(best_model_wts)
    return model
