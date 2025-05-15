import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, is_conv=True, use_dropout=False, dropout_rate=0.3):
        super(Conv_block, self).__init__()
        self.pool_op = nn.AvgPool1d(2) if is_conv else nn.Upsample(scale_factor=2, mode='linear')
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1)
        self.activation = nn.GELU()
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.pool_op(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channels=6, latent_size=16):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(in_channels, eps=0.001, momentum=0.99)

        # Layer 1: now from 6 → 64
        self.conv_block_1 = Conv_block(in_channels, 64, kernel_size=21, padding=10, #use constants to drop out rates
                                       stride=1, use_dropout=True, dropout_rate=0.2)
        # Layer 2
        self.conv_block_2 = Conv_block(64, 64, kernel_size=15, padding=7,
                                       stride=2, use_dropout=True, dropout_rate=0.2)
        # Layer 3
        self.conv_block_3 = Conv_block(64, 128, kernel_size=11, padding=5,
                                       stride=2, use_dropout=True, dropout_rate=0.2)
        # Layer 4
        self.conv_block_4 = Conv_block(128, 128, kernel_size=7, padding=3,
                                       stride=2, use_dropout=True, dropout_rate=0.2)

        # Adapt layer to fixed size 4
        self.adapt_pool = nn.AdaptiveAvgPool1d(4)

        # FC heads
        self.encode_mean   = nn.Linear(4 * 128, latent_size)
        self.encode_logvar = nn.Linear(4 * 128, latent_size)

    def forward(self, x):
        # x: (B, L, 6) → (B, 6, L)
        x = x.permute(0,2,1)
        x = self.bn0(x)

        x1 = self.conv_block_1(x)
        x2 = self.conv_block_2(x1)
        x3 = self.conv_block_3(x2)
        x4 = self.conv_block_4(x3)

        x_pooled = self.adapt_pool(x4)             # (B,128,4)
        x_flat   = x_pooled.view(x_pooled.size(0), -1)  # (B,4*128)

        mean    = self.encode_mean(x_flat)
        logvar  = self.encode_logvar(x_flat)
        return mean, logvar, x4


class Decoder(nn.Module):
    def __init__(self, length, in_channels=128, latent_size=16, out_channels= 128):
        super().__init__()
        self.length = length

        # adapt only to channels, not a fixed time‐axis
        self.adapt_nn = nn.Linear(latent_size, in_channels)
        self.relu     = nn.ReLU()
        self.tanh     = nn.Tanh()

        # Mirror the encoder blocks in reverse
        self.deconv_block_1 = Conv_block(in_channels * 2,
                                         in_channels,
                                         kernel_size=7,
                                         padding=3,
                                         is_conv=False)
        self.deconv_block_2 = Conv_block(in_channels, in_channels // 2,
                                         kernel_size=11,
                                         padding=5,
                                         is_conv=False)
        self.deconv_block_3 = Conv_block(in_channels // 2, in_channels // 2,
                                         kernel_size=15,
                                         padding=7,
                                         is_conv=False)
        self.deconv_block_4 = Conv_block(in_channels // 2, in_channels // 4,
                                         kernel_size=21,
                                         padding=10,
                                         is_conv=False)

        # getting desired size
        self.decode_conv = nn.Conv1d(in_channels // 4, out_channels,
                                     kernel_size=21, padding=10)

    def forward(self, z, skip_connection):
        # z → (B, in_channels * (L//16))
        x = self.relu(self.adapt_nn(z))
        x = x.unsqueeze(-1)
        target_len = skip_connection.size(2)
        x = F.interpolate(x,
                          size=target_len,
                          mode='linear',
                          align_corners=True)

        # concatenate skip features
        x = torch.cat([x, skip_connection], dim=1)

        # run through deconv blocks
        x = self.deconv_block_1(x)
        x = self.deconv_block_2(x)
        x = self.deconv_block_3(x)
        x = self.deconv_block_4(x)

        # upsample back to original length
        x = F.interpolate(x, size=self.length, mode="linear", align_corners=True)
        x = self.decode_conv(x)
        return self.tanh(x)


class VAE(nn.Module):
    def __init__(self, in_channels, length, latent_size=16, encoder_out_channels=128):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels,
                               latent_size=latent_size)
        self.decoder = Decoder(length,
                               in_channels=encoder_out_channels,
                               latent_size=latent_size, out_channels=in_channels)

    def reparameterize(self, mean, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std    = torch.exp(0.5 * logvar).clamp(1e-5, 10.0)
        eps    = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        # x: (B, L, C) → feed to encoder
        mean, logvar, skip_connection = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decoder(z, skip_connection)
        return recon, mean, logvar

def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)




