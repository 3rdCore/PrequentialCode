import torch


def torch_pca(x: torch.FloatTensor, center: bool = True, percent: bool = False):
    n, _ = x.shape
    # center points along axes
    if center:
        x = x - x.mean(dim=0)
    # perform singular value decomposition
    _, s, v = torch.linalg.svd(x)
    # extract components
    components = v.T
    explained_variance = torch.mul(s, s) / (n - 1)
    if percent:
        explained_variance = explained_variance / explained_variance.sum()
    return components, explained_variance
