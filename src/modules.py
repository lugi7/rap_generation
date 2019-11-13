import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

class GlobalPositionContainer(nn.Module):
    def __init__(self, emb_dim=128, milestones=[256, 512]):
        super().__init__()

        self.milestones = milestones
        self.seq_len = milestones[-1]
        self.emb_dim = emb_dim

        milestone_diff = milestones[1] - milestones[0]
        self.register_buffer('interp', self._get_interpolation(milestone_diff))

        Rc, Rd = self._get_params()
        self.Rc = nn.Parameter(Rc)
        self.Rd = nn.Parameter(Rd)



    def _get_interpolation(self, steps):
        return torch.arange(1, steps + 1).view(1, -1, 1).float() / steps

    def _get_params(self):
        start = torch.randn((1, 1, self.emb_dim)) * 0.2
        end = torch.randn((1, 1, self.emb_dim)) * 0.2

        rang = torch.arange(self.seq_len).view(1, -1, 1).float() / self.seq_len
        interpolated = (1. - rang) * start + rang * end
        interpolated = F.normalize(interpolated, p=2, dim=-1)

        return interpolated[:, :self.milestones[0]], interpolated[:, -1:]

    def forward(self):
        start = self.Rc[:, -1:]
        end = self.Rd

        interpolated = (1. - self.interp) * start + self.interp * end
        stack = torch.cat([self.Rc, interpolated], dim=1)
        normalized = F.normalize(stack, p=2, dim=-1)
        return torch.flip(normalized, dims=[1])


class DoubleAttentionLayer(nn.Module):
    def __init__(self, emb_dim=512, gp_dim=128, n_heads=8, last_layer=False):
        self.emb_dim = emb_dim
        self.n_heads = 8
        self.head_dim = emb_dim // n_heads
        self.last_layer = last_layer

        super().__init__()

        if not last_layer:
            self.Wq = nn.Linear(emb_dim, emb_dim)

        self.Wkv = nn.Linear(emb_dim, emb_dim * 2)
        self.Wp = nn.Linear(gp_dim, emb_dim)

        self.Bfc = nn.Parameter(0.2 * torch.randn(1, 1, emb_dim))
        self.Bfp = nn.Parameter(0.2 * torch.randn(1, 1, emb_dim))

        self.Wu = nn.Linear(emb_dim, emb_dim)
        self.Buc = nn.Parameter(0.2 * torch.randn(1, 1, emb_dim))
        self.Bup = nn.Parameter(0.2 * torch.randn(1, 1, emb_dim))

        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False)

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        return x

    def _get_attention(self, q, k, v, Bc, Bp, gpe, mask):
        qc = q + Bc
        qc = qc.view(qc.shape[:-1] + (self.n_heads, self.head_dim))
        qc = torch.tanh(qc)
        k = k.view(k.shape[:-1] + (self.n_heads, self.head_dim))
        k = torch.tanh(k)
        context_attention = torch.einsum('bjnd,bknd->bjkn', qc, k)

        # position attention
        qp = q + Bp
        qp = torch.tanh(qp)
        qp = qp.view(qp.shape[:-1] + (self.n_heads, self.head_dim))
        kp = self.Wp(gpe)
        kp = kp.view(kp.shape[:-1] + (self.n_heads, self.head_dim))
        kp = torch.tanh(kp)
        position_attention = torch.einsum('bjnd,bknd->bjkn', qp, kp)
        pad = context_attention.shape[2] - position_attention.shape[2]
        position_attention = F.pad(position_attention, (0, 0, pad, 0), mode='replicate')
        position_attention = self._rel_shift(position_attention)

        attention = (context_attention + position_attention) / self.head_dim ** 0.5
        attention -= 1e30 * F.pad(mask, (0, 0, pad, 0))
        attention = F.softmax(attention, dim=2)  # (bs, q_len, k_len, n_heads)

        v = v.view(v.shape[:-1] + (self.n_heads, self.head_dim))
        v = torch.tanh(v)  # (bs, k_len, n_head, head_dim)

        add = torch.einsum('bjkn,bknd->bjnd', attention, v)
        add = add.reshape(q.shape)

        return add

    def forward(self, u_emb, f_emb, gpe, u_mask, f_mask, first_part=True):
        if not first_part:
            f_mem = torch.cat([self.memory, f_emb], dim=1)
        else:
            f_mem = f_emb
        self.register_buffer('memory', f_emb)

        #unknown words embedding
        q = self.Wu(f_emb)
        kv = self.Wkv(f_mem)
        k, v = torch.chunk(kv, 2, dim=-1)

        u_add = self._get_attention(q, k, v, self.Buc, self.Bup, gpe, u_mask)
        u_emb += u_add
        u_emb = self.norm(u_emb)

        if not self.last_layer:
            q = self.Wq(f_emb)
            f_add = self._get_attention(q, k, v, self.Bfc, self.Bfp, gpe, f_mask)
            f_emb += f_add
            f_emb = self.norm(f_emb)

        return u_emb, f_emb

class EmbeddingModule(nn.Module):
    def __init__(self, num_tokens=16000, emb_dim=512, cutoffs=[1000, 2000, 4000, 8000, 16000], factor=2):
        super().__init__()
        self.emb_dim = emb_dim

        last_cutoff = 0

        self.embeddings = nn.ModuleList()
        self.mappings = nn.ModuleList()

        self.cutoffs = cutoffs
        for i, cutoff in enumerate(cutoffs):
            emb = nn.Embedding(num_embeddings=cutoff - last_cutoff + 1,
                               embedding_dim=emb_dim//factor**i,
                               padding_idx=cutoff - last_cutoff)
            if i:
                map_ = nn.Linear(emb_dim//factor**i, emb_dim, bias=False)
            else:
                map_ = nn.Identity()

            self.embeddings.append(emb)
            self.mappings.append(map_)

            last_cutoff = cutoff

    def get_emb_matrix(self):
        #get embedding matrix for dot softmax, (16000, emb_dim)
        for i, cutoff in enumerate(self.cutoffs):
            if not i:
                out = self.embeddings[i].weight[:-1]
            else:
                x = self.embeddings[i].weight[:-1]
                x = self.mappings[i](x)
                out = torch.cat([out, x])
        return out

    def forward(self, x):
        #input: long tensor (bs, seq_len)
        #output: float tensor: (bs, seq_len, emb_dim)
        last_cutoff = 0
        out = torch.zeros(x.shape + (self.emb_dim, ), device=x.device)

        for i, cutoff in enumerate(self.cutoffs):
            valid_inds = (x > last_cutoff) & (x < cutoff)
            pad = cutoff - last_cutoff

            query = (valid_inds * (x - last_cutoff)) + (~valid_inds * pad)
            embs = self.embeddings[i](query)
            embs = self.mappings[i](embs)

            out += embs
            last_cutoff = cutoff
        return out

class StemTokens(nn.Module):
    def __init__(self, emb_dim=512, n_artists=2820, n_parts=6, artist_dim=128, part_dim=8):
        self.artist_emb = nn.Linear(n_artists, artist_dim)
        self.artist_map = nn.Linear(artist_dim, emb_dim)

        self.part_emb = nn.Linear(n_parts, part_dim)
        self.part_map = nn.Linear(part_dim, emb_dim)

    def forward(self, artists, parts):
        artists = self.artist_emb(artists)
        artists = self.artist_map(artists)

        parts = self.part_emb(parts)
        parts = self.part_map(parts)

        stems = artists + parts

        return stems
