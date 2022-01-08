import torch
import os


def train(model, train_loader, test_loader, optimizer, loss_func, EPOCHES, PATH="./",device=torch.device("cpu"), checkpoint_interval=10):
    model.to(device)
    model.train()
    train_loss_recorder = []
    lr_recorder = []
    for epoch in range(EPOCHES):
        avg_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()  # clear all grad to avoid cumulation of grad
            output = model(data)
            loss_val = loss_func(output, target)
            avg_loss += loss_val.item()
            loss_val.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    "Train Epoch:{}/{} [{}/{} ({:.0f}%)] \t Loss: {:.6f}\r".format(
                        epoch + 1,
                        EPOCHES,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100 * (batch_idx / len(train_loader)),
                        loss_val.item(),
                    ),
                    end="",
                )

        avg_loss = avg_loss / (len(train_loader))
        train_loss_recorder.append([epoch, avg_loss])
        lr_recorder.append([epoch, optimizer.param_groups[0]["lr"]])
        if epoch % 20 == 0:
            print(
                "Train Epoch:{}/{} \t Average Loss: {:.6f}\r".format(
                    epoch+1, EPOCHES, avg_loss
                )
            )
        if (not checkpoint_interval is None) and epoch % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(PATH, "checkpoint_epoch_{}.pt".format(epoch)))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, os.path.join(PATH, "model_epoch_{}.pt".format(epoch)))
    return {"loss": {"train_loss": train_loss_recorder}, "lr": lr_recorder}
