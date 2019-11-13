import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import GlobalPositionContainer, EmbeddingModule, DoubleAttentionLayer

class DATModelTrain(nn.Module):
    def __init__(self, emb_dim=512, n_attention_layers=16):
        super().__init__()

        self.gpc = GlobalPositionContainer()
        self.embedding = EmbeddingModule()

        self.attention_layers = nn.ModuleList()
        for _ in range(n_attention_layers - 1):
            self.attention_layers.append(DoubleAttentionLayer())
        self.attention_layers.append(DoubleAttentionLayer(last_layer=True))

        self.register_buffer('u_mask', torch.triu(torch.ones((1, emb_dim, emb_dim, 1))))
        self.register_buffer('f_mask', torch.triu(torch.ones((1, emb_dim, emb_dim, 1)), 1))

    def forward(self, x):
        pass