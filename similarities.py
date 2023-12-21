import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearSimilarity(nn.Module):
    """
    Bilinear similarity layer
    """

    __constants__ = ["features"]
    n_features: int
    weight: torch.Tensor

    def __init__(self, n_features: int, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(BilinearSimilarity, self).__init__()
        self.n_features = n_features

        self.weight = nn.Parameter(
            torch.empty((n_features, n_features), **factory_kwargs)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        if len(input1.shape) == 3:
            input1 = input1.reshape(input1.shape[0], -1)

        if len(input2.shape) == 3:
            input2 = input2.reshape(input2.shape[0], -1)

        projection = torch.matmul(self.weight, input2.t())
        return torch.matmul(input1, projection)

    def extra_repr(self) -> str:
        return "features={}".format(self.features)


class InfoNCE(nn.Module):
    """
    Cretits: https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py

    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction="mean", negative_mode="unpaired"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(
        self, query, positive_key, negative_keys=None, labels=None, output="loss"
    ):
        return info_nce(
            query,
            positive_key,
            negative_keys,
            labels,
            temperature=self.temperature,
            reduction=self.reduction,
            negative_mode=self.negative_mode,
            output=output,
        )


def info_nce(
    query,
    positive_key,
    negative_keys=None,
    labels=None,
    temperature=0.1,
    reduction="mean",
    negative_mode="unpaired",
    output="loss",
):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError("<query> must have 2 dimensions.")
    if positive_key.dim() != 2:
        raise ValueError("<positive_key> must have 2 dimensions.")
    if negative_keys is not None:
        if negative_mode == "unpaired" and negative_keys.dim() != 2:
            raise ValueError(
                "<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'."
            )
        if negative_mode == "paired" and negative_keys.dim() != 3:
            raise ValueError(
                "<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'."
            )

    # Check matching number of samples.
    # if len(query) != len(positive_key):
    #     raise ValueError(
    #         "<query> and <positive_key> must have the same number of samples."
    #     )
    if negative_keys is not None:
        if negative_mode == "paired" and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>."
            )

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError(
            "Vectors of <query> and <positive_key> should have the same number of components."
        )
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError(
                "Vectors of <query> and <negative_keys> should have the same number of components."
            )

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == "unpaired":
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == "paired":
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal

    if output == "logits":
        return logits

    if not labels:
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
