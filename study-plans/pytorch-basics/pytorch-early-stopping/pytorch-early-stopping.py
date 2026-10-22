import torch
import torch.nn as nn

def train_with_early_stopping(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    val_losses = []
    train_losses = []
    stopped = max_epochs
    pcount = 0

    best_vl = float("inf")
    
    for epoch in range(max_epochs):
        
        epoch_train_loss = []
        epoch_val_loss = []

        model.train()
        
        for input, label in train_loader:
            optimizer.zero_grad()

            output = model(input)
            loss = criterion(output, label)

            loss.backward()

            epoch_train_loss.append(loss.item())
            
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for val_i, val_l in val_loader:
                val_out = model(val_i)
                val_loss = criterion(val_out, val_l)
                epoch_val_loss.append(val_loss.item())

        tl = sum(epoch_train_loss) / len(train_loader)
        vl = sum(epoch_val_loss) / len(val_loader)

        train_losses.append(tl)
        val_losses.append(vl)

        if vl >= best_vl:
            pcount += 1
        else:
            best_vl = vl
            pcount = 0

        if pcount >= patience:
            stopped = epoch + 1
            break
        
        

    return {
        "train_losses" : train_losses,
        "val_losses"  : val_losses,
        "stopped_epoch" : stopped
    }
    
    pass
