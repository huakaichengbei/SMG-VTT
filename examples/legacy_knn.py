"""Directly use the original uploaded gcn_lib dynamic KNN implementation."""
import torch
from smgvtt.graph.adapters.legacy_vig_adapter import LegacyDilatedKNN

if __name__=='__main__':
    x=torch.randn(2,128,256)
    neighbors=LegacyDilatedKNN(k=9,dilation=2)(x)
    print('Original gcn_lib KNN neighbors',neighbors.shape)
