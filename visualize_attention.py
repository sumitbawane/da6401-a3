#!/usr/bin/env python3
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. DATA
eng_train  = ['namaste','bharat','shanti','yoga','guru','diwali','hindi','chai','krishna']
hin_train  = ['नमस्ते','भारत','शांति','योग','गुरु','दिवाली','हिंदी','चाय','कृष्णा']
test_words = ['ramayana','gurukul','festival','rath','mantra']

PAD, SOS, EOS = '<pad>', '<sos>', '<eos>'

def build_vocab(words):
    chars = set(''.join(words))
    vocab = [PAD, SOS, EOS] + sorted(chars)
    idx = {c: i for i, c in enumerate(vocab)}
    inv = {i: c for c, i in idx.items()}
    return idx, inv

src_idx, src_inv = build_vocab(eng_train + test_words)
tgt_idx, tgt_inv = build_vocab(hin_train)

def tensorize(word, idx_map):
    # [SOS] + chars + [EOS]
    seq = [idx_map[SOS]] + [idx_map.get(c, idx_map[PAD]) for c in word] + [idx_map[EOS]]
    return torch.tensor(seq, dtype=torch.long)

pairs = list(zip(eng_train, hin_train))

# 2. MODEL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, in_dim, emb_dim, hid_dim):
        super().__init__()
        self.emb = nn.Embedding(in_dim, emb_dim, padding_idx=src_idx[PAD])
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
    def forward(self, x):
        return self.rnn(self.emb(x))

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W1 = nn.Linear(hid_dim, hid_dim)
        self.W2 = nn.Linear(hid_dim, hid_dim)
        self.V  = nn.Linear(hid_dim,  1)
    def forward(self, h, enc_out):
        src_len = enc_out.size(1)
        # h: [1, batch, hid_dim] → [batch, 1, hid_dim] → repeat → [batch, src_len, hid_dim]
        h_rep = h.permute(1,0,2).repeat(1,src_len,1)
        energy = torch.tanh(self.W1(enc_out) + self.W2(h_rep))
        scores = self.V(energy).squeeze(2)          # [batch, src_len]
        return torch.softmax(scores, dim=1)

class Decoder(nn.Module):
    def __init__(self, out_dim, emb_dim, hid_dim, attn):
        super().__init__()
        self.emb    = nn.Embedding(out_dim, emb_dim, padding_idx=tgt_idx[PAD])
        self.rnn    = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim*2, out_dim)
        self.attn   = attn

    def forward(self, x, h, enc_out):
        x   = x.unsqueeze(1)                          # [batch,1]
        emb = self.emb(x)                             # [batch,1,emb_dim]
        a   = self.attn(h, enc_out)                   # [batch,src_len]
        ctx = torch.bmm(a.unsqueeze(1), enc_out)       # [batch,1,hid_dim]
        out, h2 = self.rnn(torch.cat([emb, ctx], dim=2), h)
        out = out.squeeze(1)                          # [batch,hid_dim]
        ctx = ctx.squeeze(1)                          # [batch,hid_dim]
        return self.fc_out(torch.cat([out,ctx], dim=1)), h2, a

# Hyperparameters
EMB_DIM = 32
HID_DIM = 64
LR      = 1e-3
EPOCHS  = 300

enc   = Encoder(len(src_idx), EMB_DIM, HID_DIM).to(device)
attn  = Attention(HID_DIM).to(device)
dec   = Decoder(len(tgt_idx), EMB_DIM, HID_DIM, attn).to(device)

opt   = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
crit  = nn.CrossEntropyLoss(ignore_index=tgt_idx[PAD])

# 3. TRAIN
for epoch in range(1, EPOCHS+1):
    epoch_loss = 0
    for sw, tw in pairs:
        src_seq = tensorize(sw, src_idx).unsqueeze(0).to(device)
        tgt_seq = tensorize(tw, tgt_idx).to(device)

        enc_out, hidden = enc(src_seq)
        inp = torch.tensor([tgt_idx[SOS]], device=device)
        loss = 0

        for t in range(1, tgt_seq.size(0)):
            pred, hidden, _ = dec(inp, hidden, enc_out)
            loss += crit(pred, tgt_seq[t].unsqueeze(0))
            inp = tgt_seq[t].unsqueeze(0)

        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{EPOCHS} — Avg Loss: {epoch_loss/len(pairs):.4f}")

# 4. DECODE & PRINT ATTENTION
print("\n=== Attention Alignments ===\n")
for test_w in test_words:
    src_seq, hidden = enc(tensorize(test_w, src_idx).unsqueeze(0).to(device))
    inp = torch.tensor([tgt_idx[SOS]], device=device)

    attn_mat, out_idxs = [], []
    for _ in range(2 * len(test_w)):
        pred, hidden, a = dec(inp, hidden, src_seq)
        idx = pred.argmax(1).item()
        out_idxs.append(idx)
        attn_mat.append(a.detach().cpu().numpy()[0])
        inp = torch.tensor([idx], device=device)
        if idx == tgt_idx[EOS]:
            break

    pred_chars = [tgt_inv[i] for i in out_idxs]
    attn_mat   = np.array(attn_mat)
    src_chars  = list(test_w)
    top2_idxs  = np.argsort(-attn_mat[:,1:1+len(src_chars)], axis=1)[:,:2]

    print(f"--- Translating: '{test_w}' ---")
    for i, out_ch in enumerate(pred_chars):
        if out_ch == EOS:
            break
        i1, i2 = top2_idxs[i]
        # highlight two positions
        highlighted = [
            c.upper() if j in (i1,i2) else c.lower()
            for j,c in enumerate(src_chars)
        ]
        disp = ' '.join(highlighted)
        c1, c2 = src_chars[i1], src_chars[i2]
        print(f"While translating '{out_ch}' → paid attention to '{c1}','{c2}'")
        print(f"   → {disp}\n")
