"""Test the low-rank approximation function and ModuleLowRank class."""

import numpy as np
import torch
import torch.nn as nn

from eigen_guess.low_rank import (
    ModuleLowRank,
    compute_low_rank_matrices,
    create_low_rank_modules,
    low_rank_approximate,
    svd_decomposition,
)

from eigen_guess.low_rank_org import low_rank_approximate as lra , ModuleLowRank


def test_svd_decomposition():
    """Test the svd_decomposition function."""
    mat_org = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    u, s, vh = svd_decomposition(mat_org)
    assert u.shape == (2, 2)
    assert vh.shape == (2, 2)
    assert len(s) == 2

def test_compute_low_rank_matrices():
    """Test the compute_low_rank_matrices function."""
    u = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = np.array([1.0, 0.5])
    vh = np.array([[1.0, 0.0], [0.0, 1.0]])
    rank = 2

    mat_q, mat_r = compute_low_rank_matrices(u, s, vh, rank)
    assert mat_q.shape == (2, rank)
    assert mat_r.shape == (rank, 2)

def test_low_rank_approximate():
    """Test the low_rank_approximate function."""
    mat = torch.randn(100, 100)
    rank = 25
    result = low_rank_approximate(mat, rank)
    mat_q_org, mat_r_org = lra(mat, rank)['mat_q'], lra(mat, rank)['mat_r']
    print("mat_q_org", mat_q_org.shape)
    print("mat_r_org", mat_r_org.shape)
    assert 'mat_q' in result
    assert 'mat_r' in result
    assert 'error' in result
    assert result['mat_q'].shape == mat_q_org.shape
    assert result['mat_r'].shape == mat_r_org.shape


def test_create_low_rank_modules():
    """Test the create_low_rank_modules function."""
    shape = (100, 100)
    rank = 25
    bias = True
    module_l, module_r = create_low_rank_modules(shape, rank, bias)
    print("module_l in_features", module_l.in_features)
    print("module_l out_features", module_l.out_features)
    print("module_r in_features", module_r.in_features)
    print("module_r out_features", module_r.out_features)
    assert isinstance(module_l, nn.Linear)
    assert isinstance(module_r, nn.Linear)
    assert module_l.out_features == rank
    assert module_r.in_features == rank
    assert module_r.out_features == shape[1]


def create_sample_model():
    """Create a sample neural network with linear layers."""
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    return model

def test_module_low_rank():
    """Test the ModuleLowRank class."""
    model = create_sample_model()
    low_rank_module = ModuleLowRank(compress_ratio=3, is_approximate=True)
    modified_model = low_rank_module(model)

    assert isinstance(modified_model, nn.Sequential)
    assert len(modified_model) == 5
    assert isinstance(modified_model[0], nn.Sequential)
    assert isinstance(modified_model[1], nn.ReLU)
    assert isinstance(modified_model[2], nn.Sequential)
    assert isinstance(modified_model[3], nn.ReLU)
    assert isinstance(modified_model[4], nn.Sequential)

    for i in (0, 2, 4):
        assert isinstance(modified_model[i][0], nn.Linear)
        assert isinstance(modified_model[i][1], nn.Linear)

    for i in (1, 3):
        assert isinstance(modified_model[i], nn.ReLU)

    rank1 = model[0][0].in_features * model[0][1].out_features // (3 * (
        model[0][0].in_features + model[0][1].out_features)
    )

    assert modified_model[0][0].in_features == 64
    assert modified_model[0][0].out_features == int(rank1)
    assert modified_model[0][1].in_features == int(rank1)
    assert modified_model[0][1].out_features == 128

    rank2 = model[2][0].in_features * model[2][1].out_features // (3 * (
        model[2][0].in_features + model[2][1].out_features)
    )

    assert modified_model[2][0].in_features == 128
    assert modified_model[2][0].out_features == int(rank2)
    assert modified_model[2][1].in_features == int(rank2)
    assert modified_model[2][1].out_features == 256

    rank3 = model[4][0].in_features *model[4][1].out_features // (3 * (
        model[4][0].in_features + model[4][1].out_features)
    )

    assert modified_model[4][0].in_features == 256
    assert modified_model[4][0].out_features == int(rank3)
    assert modified_model[4][1].in_features == int(rank3)
    assert modified_model[4][1].out_features == 10

