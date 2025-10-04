import torch, torch.nn as nn
import math

class ConvSubsample(nn.Module):
    def __init__(self, in_dim=512, out_dim=256, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=5, stride=stride, padding=2),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
    def forward(self, x):  # x: (B, T, D)
        x = x.transpose(1, 2)         # (B, D, T)
        x = self.net(x)               # (B, C, T')
        x = x.transpose(1, 2)         # (B, T', C)
        return x

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion=4, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, expansion*d_model),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(expansion*d_model, d_model),
            nn.Dropout(p),
        )
    def forward(self, x):
        return self.net(x)

class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=15, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, 2*d_model)
        self.glu = nn.GLU()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(p)
    def forward(self, x):  # (B,T,D)
        y = self.ln(x)
        y = self.pw1(y)
        y = self.glu(y)               # (B,T,D)
        y = y.transpose(1, 2)
        y = self.dw(y)
        y = self.bn(y)
        y = self.swish(y)
        y = self.pw2(y)
        y = y.transpose(1, 2)
        y = self.dropout(y)
        return y

class MHSA(nn.Module):
    def __init__(self, d_model, nhead=4, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=p, batch_first=True)
        self.dropout = nn.Dropout(p)
    def forward(self, x, key_padding_mask=None):
        y = self.ln(x)
        y, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout(y)

class ConformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, ff_ratio=0.5, p=0.1, conv_kernel=15):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, expansion=int(1/ff_ratio), p=p)
        self.mhsa = MHSA(d_model, nhead=nhead, p=p)
        self.conv = ConvModule(d_model, kernel_size=conv_kernel, p=p)
        self.ff2 = FeedForwardModule(d_model, expansion=int(1/ff_ratio), p=p)
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x, key_padding_mask=None):
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x, key_padding_mask=key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)

class ConformerEncoder(nn.Module):
    def __init__(self, in_dim=512, d_model=256, num_blocks=12, nhead=4, p=0.1, subsample=2):
        super().__init__()
        self.sub = ConvSubsample(in_dim, d_model, stride=subsample)
        self.blocks = nn.ModuleList([ConformerBlock(d_model, nhead=nhead, p=p) for _ in range(num_blocks)])
    def forward(self, x, x_lens):
        # x: (B,T,D)
        x = self.sub(x)  # (B,T',d_model)
        x_lens = (x_lens // 2).clamp(min=1)
        # build key_padding_mask
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device)[None, :].expand(B, T) >= x_lens[:, None]
        for blk in self.blocks:
            x = blk(x, key_padding_mask=mask)
        return x, x_lens

class RNNTPredictor(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden=512, layers=2, p=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden, num_layers=layers, batch_first=True, dropout=p if layers>1 else 0.0)
    def forward(self, y):  # (B,U)
        e = self.embed(y)
        out, _ = self.rnn(e)
        return out  # (B,U,H)

class RNNTJoint(nn.Module):
    def __init__(self, enc_dim, pred_dim, joint_dim, vocab_size):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(enc_dim + pred_dim, joint_dim),
            nn.SiLU(),
            nn.Linear(joint_dim, vocab_size)
        )
    def forward(self, enc, pred):
        # enc: (B,T,De), pred: (B,U,Dp) -> (B,T,U,V)
        B,T,De = enc.shape
        B,U,Dp = pred.shape
        e = enc.unsqueeze(2).expand(B,T,U,De)
        p = pred.unsqueeze(1).expand(B,T,U,Dp)
        z = torch.cat([e,p], dim=-1)
        logits = self.ff(z)
        return logits

class ConformerRNNT(nn.Module):
    def __init__(self, in_dim=512, d_model=256, num_blocks=12, nhead=4, p=0.1,
                 vocab_size=2000, pred_hidden=512, pred_layers=2, joint_dim=512):
        super().__init__()
        self.encoder = ConformerEncoder(in_dim, d_model, num_blocks, nhead, p)
        self.predictor = RNNTPredictor(vocab_size, emb_dim=d_model, hidden=pred_hidden, layers=pred_layers, p=p)
        self.joint = RNNTJoint(d_model, pred_hidden, joint_dim, vocab_size)
        # auxiliary CTC head
        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward_encoder(self, x, x_lens):
        return self.encoder(x, x_lens)

    def forward(self, x, x_lens, y_in):
        enc, enc_lens = self.encoder(x, x_lens)
        pred = self.predictor(y_in)
        joint_logits = self.joint(enc, pred)  # (B,T,U,V)
        ctc_logits = self.ctc_head(enc)       # (B,T,V)
        return joint_logits, ctc_logits, enc_lens