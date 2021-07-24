import torch
import torch.nn as nn
from complex_models.sublayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model,
                 d_ffn,
                 n_head,
                 dropout=0.1,
                 continue_complex=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(
                        d_model=d_model,
                        n_head=n_head,
                        attn_drop=dropout,
                        continue_complex=continue_complex
                        )
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout_1 = nn.Dropout(dropout)

        self.pos_ffn = PositionwiseFeedForward(
                        d_model=d_model,
                        d_ffn=d_ffn,
                        dropout=dropout
                        )
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x_real, x_phase, mask=None):
        # x: [b, seq_len, d_model]
        residual_1_real = x_real
        residual_1_phase = x_phase
        x_real, x_phase = self.self_attn(
                                [x_real, x_real, x_real], 
                                [x_phase, x_phase, x_phase], 
                                mask)
        x_real = self.norm_1(residual_1_real + self.dropout_1(x_real))
        x_phase = self.norm_1(residual_1_phase + self.dropout_1(x_phase))

        residual_2_real = x_real
        residual_2_phase = x_phase
        x_real, x_phase = self.pos_ffn(x_real, x_phase)
        x_real = self.norm_2(residual_2_real + self.dropout_2(x_real))
        x_phase = self.norm_2(residual_2_phase + self.dropout_2(x_phase))

        return x_real, x_phase # [b, seq_len, d_model]


class DecoderLayer(nn.Module):
    def __init__(self, 
                 d_model,
                 d_ffn,
                 n_head,
                 dropout=0.1,
                 continue_complex=False):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(
                            d_model=d_model,
                            n_head=n_head,
                            attn_drop=dropout,
                            continue_complex=continue_complex
                            )
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout_1 = nn.Dropout(dropout)

        self.enc_dec_attn = MultiHeadAttention(
                            d_model=d_model,
                            n_head=n_head,
                            attn_drop=dropout,
                            continue_complex=False
                            )
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout_2 = nn.Dropout(dropout)

        self.pos_ffn = PositionwiseFeedForward(
                            d_model=d_model,
                            d_ffn=d_ffn,
                            dropout=dropout
                            )
        self.norm_3 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x_real, x_phase, enc_out_real, enc_put_phase, 
                dec_enc_attn_mask=None, slf_attn_mask=None):
        
        residual_1_real = x_real
        residual_1_phase = x_phase
        x_real, x_phase = self.masked_self_attn([x_real, x_real, x_real], 
                                  [x_phase, x_phase, x_phase], 
                                  slf_attn_mask)
        x_real = self.norm_1(residual_1_real + self.dropout_1(x_real))
        x_phase = self.norm_1(residual_1_phase + self.dropout_1(x_phase))
        
        residual_2_real = x_real
        residual_2_phase = x_phase
        x_real, x_phase = self.enc_dec_attn([x_real, enc_out_real, enc_out_real],
                              [x_phase, enc_put_phase, enc_put_phase],
                              dec_enc_attn_mask)
        x_real = self.norm_2(residual_2_real + self.dropout_2(x_real))
        x_phase = self.norm_2(residual_2_phase + self.dropout_2(x_phase))

        residual_3_real = x_real
        residual_3_phase = x_phase
        x_real, x_phase = self.pos_ffn(x_real, x_phase)
        x_real = self.norm_3(residual_3_real + self.dropout_3(x_real))
        x_phase = self.norm_3(residual_3_phase + self.dropout_3(x_phase))

        return x_real, x_phase # [b, seq_len, d_model]


if __name__ == '__main__':
    x_real = torch.randn(1, 10, 512)
    x_phase = torch.randn(1, 10, 512)
    encoderlayer = EncoderLayer(d_model=512, d_ffn=2048, n_head=8)
    ecd_out_real, ecd_out_phase = encoderlayer(x_real, x_phase)
    print(ecd_out_phase.size())

    decoderlayer = DecoderLayer(d_model=512, d_ffn=2048, n_head=8)
    dcd_out = decoderlayer(x_real, x_phase, ecd_out_real, ecd_out_phase)
    print(dcd_out[0].size())
