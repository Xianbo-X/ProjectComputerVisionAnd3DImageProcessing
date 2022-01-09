import torch


class TestTools:
    registered_service = {}

    def __init__(self, data, targets) -> None:
        pass

    @classmethod
    def accuracy_rate(cls, model, data, targets):
        cls.registered_service["accuracy"] = cls.accuracy_rate  # not work as hope
        model.eval()
        with torch.no_grad():
            result = model.predict(data.reshape(-1, 1, 28, 28).float())
        return torch.sum(result == targets).item() / len(targets)

    @classmethod
    def loss(cls, model, loss_func, data, targets):
        cls.registered_service["loss"] = cls.loss
        model.eval()
        with torch.no_grad():
            data = data.reshape(-1, 1, 28, 28).float()
            output = model(data)
            loss_val = loss_func(output, targets)
        return loss_val.item()
