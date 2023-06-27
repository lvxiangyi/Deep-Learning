import re
import math
import importlib
import copy
import spacy
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
class Embedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):

    def __init__(self, d_model, max_seq_len=1200):
        super().__init__()
        self.d_model = d_model

        # 根据pos和i创建一个常量pe矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 让 embeddings vector 相对大一些
        # x = x * math.sqrt(self.d_model)
        x = x * 4
        # 增加位置常量到 embedding 中
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].cuda()

        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # mask掉那些为了padding长度增加的token，让其通过softmax计算后为0
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class NormLayer(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):

    def __init__(self, d_model, heads, dropout):
        super().__init__()

        # self attention
        self.sa = MultiHeadAttention(heads, d_model)

        # normalization layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # feed forward network
        self.feed_forward = FeedForward(d_model)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # self attention
        x = self.norm1(x)
        x = self.sa(x, x, x)
        x = self.dropout(x) + x

        # feed forward
        y = self.norm2(x)
        y = self.feed_forward(y)
        y = self.dropout(y) + y

        return y
class DecoderLayer(nn.Module):

    def __init__(self, d_model, heads, dropout):
        super().__init__()

        # self attention layer
        self.sa = MultiHeadAttention(heads, d_model)

        # normalize, before and after self-attention layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # feed forward network
        self.feed_forward = FeedForward(d_model)

    def forward(self, x):
        # self attention
        x = self.norm1(x)
        x = self.sa(x, x, x)
        x = self.dropout(x) + x

        # feed forward
        y = self.norm2(x)
        y = self.feed_forward(y)
        y = self.dropout(y) + y
        return y

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):

    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = NormLayer(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):

    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = NormLayer(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(self, d_model, heads, N, dropout):
        super().__init__()

        # 编码器的每一层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads, dropout) for _ in range(N)])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        # src = [batch_size,src_len,d_model]
        for layer in self.layers:
            # 编码器里的每一层
            src = layer(src, mask)

        # 出自最后一层后应用层标准化
        src = self.norm(src)

        return src


class TransModel(nn.Module):

    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoding = PositionalEncoder(d_model)

        self.transformer = Transformer(d_model, heads, N, dropout)

        self.fc = nn.Linear(d_model, 2)  # 2分类:正面/负面评论

    def forward(self, x,lengths):

        # Embeddings + Positional Encoding

        out = self.embedding(x)  # 嵌入
        # out = nn.utils.rnn.pack_padded_sequence(
        #     out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        # )
        x = self.pos_encoding(out)  # 位置编码

        # Transformer Encoding
        x = self.transformer(x)

        # Output Layer
        output = self.fc(x[:, 0, :])  # 取第一词嵌入作为分类输入

        return output