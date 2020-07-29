import torch
import numpy as np
from pytorch_isotonic import isotonic_l2


def _check_regularization(regularization):
    if regularization not in ("l2", "kl"):
        raise ValueError("'regularization' should be either 'l2' or 'kl' "
                         "but got %s." % str(regularization))


def _inv_permutation(permutation):
    """Returns inverse permutation of 'permutation'."""
    inv_permutation = torch.zeros(len(permutation), dtype=torch.IntTensor)
    inv_permutation[permutation] = torch.arange(len(permutation))
    return inv_permutation


def _partition(solution, eps=1e-9):
    """Returns partition corresponding to solution."""
    if len(solution) == 0:
        return []

    sizes = [1]

    for i in range(1, len(solution)):
        if abs(solution[i] - solution[i - 1]) > eps:
            sizes.append(0)
        sizes[-1] += 1

    return sizes


class Isotonic():
    """Isotonic optimization."""

    def __init__(self, input_s, input_w, regularization="l2"):
        self.input_s = input_s
        self.input_w = input_w
        self.regularization = regularization
        self.solution_ = None

    def size(self):
        return len(self.input_s)

    def compute(self):

        self.solution_ = isotonic_l2(self.input_s, self.input_w)
        return self.solution_

    def _check_computed(self):
        if self.solution_ is None:
            raise RuntimeError("Need to run compute() first.")

    def jvp(self, vector):
        self._check_computed()
        start = 0
        return_value = torch.zeros_like(self.solution_)
        for size in _partition(self.solution_):
            end = start + size
            val = torch.mean(vector[start:end])
            return_value[start:end] = val
            start = end
        return return_value

    def vjp(self, vector):
        start = 0
        return_value = torch.zeros_like(self.solution_)
        for size in _partition(self.solution_):
            end = start + size
            val = 1. / size
            return_value[start:end] = val * torch.sum(vector[start:end])
            start = end
        return return_value


class SoftSort():
    """Soft sorting."""

    def __init__(self, values, direction="ASCENDING",
                 regularization_strength=1.0, regularization="l2"):
        self.values = values
        self.sign = 1 if direction == "DESCENDING" else -1
        self.regularization_strength = regularization_strength
        _check_regularization(regularization)
        self.regularization = regularization
        self.isotonic_ = None

    def size(self):
        return len(self.values)

    def _check_computed(self):
        if self.isotonic_ is None:
            raise ValueError("Need to run compute() first.")

    def compute(self):
        size = len(self.values)
        print(size)
        input_w = torch.flip(torch.arange(start=1, end=size + 1, step=1))

        input_w = input_w / self.regularization_strength
        values = self.sign * self.values
        self.permutation_ = torch.flip(torch.argsort(values))
        
        s = values[self.permutation_]

        self.isotonic_ = Isotonic(
            input_w, s, regularization=self.regularization)
        res = self.isotonic_.compute()

        # We set s as the first argument as we want the derivatives w.r.t. s.
        self.isotonic_.s = s
        return self.sign * (input_w - res)

    def jvp(self, vector):
        self._check_computed()
        return self.isotonic_.jvp(vector[self.permutation_])

    def vjp(self, vector):
        self._check_computed()
        inv_permutation = _inv_permutation(self.permutation_)
        return self.isotonic_.vjp(vector)[inv_permutation]

direction="ASCENDING"
regularization_strength=1.0
regularization="l2"

def map_tensor(map_fn, tensor):
  return torch.stack([map_fn(tensor_i) for tensor_i in torch.unbind(tensor)])

class SS_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        ss = SoftSort(values, direction,\
            regularization_strength, regularization)
        ctx.ss = ss
        return ss.compute()
    @staticmethod
    def backward(ctx, grad_output):
        ss = ctx.ss
        return ss.vjp(grad_output)


def soft_sort_pytorch(values, direction="ASCENDING",\
    regularization_strength=1.0, regularization="l2"):
    return map_tensor(SS_Func.apply, values)

