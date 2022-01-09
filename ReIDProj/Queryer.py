import torch
import numpy as np


class QueryReID():
    def __init__(
        self, model, gallery_data, gallery_targets, dist_func="euclidean", gallery=None
    ) -> None:
        self.model = model
        self.model.eval()
        self.gallery_data = gallery_data
        self.gallery_targets = gallery_targets
        if gallery is None:
            self.init_gallery()
        else:
            self.gallery = gallery
        if dist_func.upper() == "EUCLIDEAN".upper():
            self.distance = self.euclidien_square_distance

    def init_gallery(self):
        feature_gallery = (
            self.model.feature_extraction(self.gallery_data.reshape(-1, 1, 28, 28))
            .detach()
            .numpy()
        )
        self.gallery = pd.concat(
            [
                pd.DataFrame(feature_gallery, columns=["x", "y"]),
                pd.DataFrame(self.gallery_targets, columns=["label"]),
            ],
            axis=1,
            join="inner",
        )

    def inner_distance(self, vector1, vector2):
        assert vector1.shape[1] == vector2.shape[1], (
            str(vector1.shape) + "not comply" + str(vector2.shape)
        )
        return (vector1 @ vector2.T).squeeze()

    def euclidien_square_distance(self, vector1, vector2):
        """
        Return Distance between every row vector in vector 1 and every row vector in vector2
        i.e. reuslt[i][j]=dist(vector1[i,:],vector2[j,:])
        """
        assert vector1.shape[1] == vector2.shape[1], (
            str(vector1.shape) + "not comply" + str(vector2.shape)
        )
        return np.linalg.norm(
            (feature_temp - mnist_query.gallery[["x", "y"]].to_numpy()[:, None]),
            axis=-1,
        ).T

    def query(self, data, k):
        """
        batch query
        """
        feature = self.get_query_feature(data)
        score = np.linalg.norm(
            (feature - self.gallery[["x", "y"]].to_numpy()[:, None]), axis=-1
        ).T
        return np.argsort(score, axis=1)[:, :k]

    def get_query_feature(self, data):
        return self.model.feature_extraction(data).detach().numpy().reshape(-1, 2)

    def get_gallery_feature(self, idx):
        return self.gallery[idx]

    def get_gallery_data(self, idx):
        return (self.gallery_data[idx], self.gallery_targets[idx])

    def reid_accuracy_rate(self,query_data,query_target,n_top,reduce="none"):
        if self.gallery is None: self.init_gallery()
        top_n=self.query(query_data,n_top)
        data_all,target_all=self.get_gallery_data(top_n)
        assert (target_all.shape==(len(query_target),n_top))
        true_count=(target_all==query_target.reshape(-1,1)).sum(axis=1)
        acc_rate=true_count/n_top
        if reduce.upper()=="MEAN":
            acc_rate=acc_rate.mean()
        return acc_rate

    def draw(self,top_n,idx,query_data,query_target):
        from matplotlib import pyplot as plt
        assert top_n.shape[1]==20, "only designed for top20"
        for i in range(top_n.shape[0]):
            data,targets=self.get_gallery_data(top_n[i,:])
            print("digit_{} \t sample idx {} \t TOP 20 accuracy {}".format(query_target[i],idx[i],np.sum(targets==query_target[i])/len(targets)))
            rows=4
            cols=5
            factor=3
            fig=plt.figure(figsize=(cols*factor,rows*factor))
            axes=np.array(fig.subplots(rows,cols)).ravel()
            for j in range(len(data)):
                axes[j].imshow(data[j])
                axes[j].set_title("digit {}".format(targets[j]))
                axes[j].axis("off")
            fig.tight_layout()
            fig2=plt.figure("query image{}".format(i))
            ax=fig2.subplots(1,1)
            ax.imshow(query_data[i].squeeze())
