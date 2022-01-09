import torch
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from abc import ABCMeta,abstractmethod

class Visaulizer(metaclass=ABCMeta):
    def __init__(self, dataset) -> None:
        self.dataset=dataset
    
    @abstractmethod
    def __call__(self, model:torch.nn.Module):
        pass

class VisualizationEmbeding(Visaulizer):
    def __init__(self, dataset:Dataset, random_seed=0) -> None:
        super(VisualizationEmbeding,self).__init__(dataset)
        self.dataset = dataset
        np.random.seed(random_seed)

    def sample_datas(self, sample_nums):
        self.samples = self._get_n_samples(sample_nums)

    def _get_n_samples(self, n:int)->dict:
        samples = {}
        for num in range(10):
            digit_set = torch.where(self.dataset.targets == num)[0]
            random_idx = np.random.randint(0, len(digit_set) + 1, n)
            samples[num] = digit_set[random_idx]
        return samples

    def draw_embeding_space(self, all_points):
        # Plot and show
        colors = matplotlib.cm.Paired(np.linspace(0, 1, len(all_points)))
        fig = plt.figure(figsize=(10, 10))
        ax = fig.subplots()
        for (points, color, digit) in zip(all_points, colors, range(10)):
            ax.scatter(
                [item[0] for item in points],
                [item[1] for item in points],
                color=color,
                label="digit {}".format(digit),
            )
            ax.grid(True)
            ax.legend()
        return fig

    def feature_exract_draw(self, model:torch.nn.Module):
        all_points = []
        for i in range(10):
            index = self.samples[i]
            data = self.dataset.data[index].reshape(-1, 1, 28, 28) / 255
            targets = self.dataset.targets[index]
            all_points.append(model.feature_extraction(data).detach().numpy())
        self.draw_embeding_space(all_points)
    
    def show_samples(self):
        for key,data in self.samples.items():
            fig=plt.figure()
            ax=np.array(fig.subplots(4,5)).ravel()
            for i in range(len(data)):
                ax[i].imshow(self.dataset.data[data[i]])

    def __call__(self, model):
        self.feature_exract_draw(model)