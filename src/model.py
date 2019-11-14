import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_memlab import profile

if __name__ == '__main__':
    from .modules import GlobalPositionContainer, EmbeddingModule, DoubleAttentionLayer, StemTokens
else:
    from modules import GlobalPositionContainer, EmbeddingModule, DoubleAttentionLayer, StemTokens


class DATModel(nn.Module):
    def __init__(self, emb_dim=512, n_attention_layers=16, gather_layers=[4, 8, 12, 16], seq_len=512,):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_attention_layers = n_attention_layers
        # self.n_last_softmax = n_last_softmax
        self.gather_layers = gather_layers

        self.gpc = GlobalPositionContainer()
        self.embedding = EmbeddingModule(emb_dim=emb_dim)
        self.stems = StemTokens(emb_dim=emb_dim)

        self.attention_layers = nn.ModuleList()
        for _ in range(n_attention_layers - 1):
            self.attention_layers.append(DoubleAttentionLayer(emb_dim=emb_dim))
        self.attention_layers.append(DoubleAttentionLayer(emb_dim=emb_dim, last_layer=True))

        u_mask = torch.unsqueeze(torch.triu(torch.ones((1, seq_len, seq_len))), -1)
        f_mask = torch.unsqueeze(torch.triu(torch.ones((1, seq_len, seq_len)), 1), -1)

        self.register_buffer('u_mask', u_mask)
        self.register_buffer('f_mask', f_mask)

    def forward(self, tokens, parts, artists, first_part=True):
        u_emb = self.stems(parts, artists)
        f_emb = self.embedding(tokens) + u_emb
        gpe = self.gpc()

        to_softmax = []
        for i, att in enumerate(self.attention_layers):
            u_emb, f_emb = att(u_emb, f_emb, gpe, self.u_mask, self.f_mask, first_part)
            if i + 1 in self.gather_layers:
                to_softmax.append(u_emb)
        to_softmax = torch.stack(to_softmax) #(4, 32, 512, 512)
        emb_matrix = self.embedding.get_emb_matrix()
        logits = torch.einsum('nbjd,vd->bjv', to_softmax, emb_matrix)
        logits = logits / (to_softmax.shape[0] * (self.emb_dim ** 0.5))

        return logits

    # def infer_step(self, tokens, parts, artits):
    #   pass