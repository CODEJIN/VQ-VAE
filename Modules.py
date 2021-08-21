import torch
from argparse import Namespace

class VQVAE(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace) -> None:
        super().__init__()
        self.hp = hyper_parameters

        self.encoder = Encoder(self.hp)
        self.vqvae = Vector_Quantizer(
            num_embeddings= self.hp.VQVAE.Num_Quantizing,
            embedding_size= self.hp.Encoder.Channels,
            )
        self.decoder = Decoder(self.hp)

    def forward(self, x: torch.Tensor):
        encodings = self.encoder(x)
        quantizeds, vqvae_losses, perplexities = self.vqvae(encodings)
        reconstructions = self.decoder(quantizeds)

        return reconstructions, vqvae_losses, perplexities


class Encoder(torch.nn.Sequential):
    def __init__(self, hyper_parameters: Namespace) -> None:
        super().__init__()
        self.hp = hyper_parameters

        self.conv = torch.nn.Sequential()
        self.add_module('Conv_0', Conv2d(
            in_channels= 3,
            out_channels= self.hp.Encoder.Channels // 2,
            kernel_size= 4,
            stride= 2,
            padding= 1,
            w_init_gain= 'relu'
            ))
        self.add_module('ReLU_0', torch.nn.ReLU())
        self.add_module('Conv_1', Conv2d(
            in_channels= self.hp.Encoder.Channels // 2,
            out_channels= self.hp.Encoder.Channels,
            kernel_size= 4,
            stride= 2,
            padding= 1,
            w_init_gain= 'relu'
            ))
        self.add_module('ReLU_1', torch.nn.ReLU())
        self.add_module('Conv_2', Conv2d(
            in_channels= self.hp.Encoder.Channels,
            out_channels= self.hp.Encoder.Channels,
            kernel_size= 3,
            padding= 1,
            w_init_gain= 'relu'
            ))
        self.add_module('ReLU_2', torch.nn.ReLU())

        for index in range(self.hp.Encoder.Residual.Stack):
            self.add_module('CFN_{}'.format(index), CFN(
                channels= self.hp.Encoder.Channels,
                calc_channels= self.hp.Encoder.Residual.Calc_Channels,
                ))

    def forward(self, x: torch.Tensor):
        return super().forward(x)

class Decoder(torch.nn.Sequential):
    def __init__(self, hyper_parameters: Namespace) -> None:
        super().__init__()
        self.hp = hyper_parameters

        self.conv = torch.nn.Sequential()
        self.add_module('Conv', Conv2d(
            in_channels= self.hp.Encoder.Channels,
            out_channels= self.hp.Decoder.Channels,
            kernel_size= 3,
            padding= 1,
            w_init_gain= 'linear'
            ))

        for index in range(self.hp.Decoder.Residual.Stack):
            self.add_module('CFN_{}'.format(index), CFN(
                channels= self.hp.Decoder.Channels,
                calc_channels= self.hp.Decoder.Residual.Calc_Channels,
                ))

        self.add_module('ConvTranspose_0', ConvTranspose2d(
            in_channels= self.hp.Decoder.Channels,
            out_channels= self.hp.Decoder.Channels // 2,
            kernel_size= 4,
            stride= 2,
            padding= 1,
            w_init_gain= 'relu'
            ))
        self.add_module('ReLU', torch.nn.ReLU())
        self.add_module('ConvTranspose_1', ConvTranspose2d(
            in_channels= self.hp.Decoder.Channels // 2,
            out_channels= 3,
            kernel_size= 4,
            stride= 2,
            padding= 1,
            w_init_gain= 'linear'
            ))

    def forward(self, x: torch.Tensor):
        return super().forward(x)

class CFN(torch.nn.Sequential):
    def __init__(
        self,
        channels: int,
        calc_channels: int
        ) -> None:
        super().__init__()

        self.add_module('Conv_0', Conv2d(
            in_channels= channels,
            out_channels= calc_channels,
            kernel_size= 3,
            padding= 1,
            bias= False,
            w_init_gain= 'relu'
            ))
        self.add_module('ReLU_0', torch.nn.ReLU())
        self.add_module('Conv_1', Conv2d(
            in_channels= calc_channels,
            out_channels= channels,
            kernel_size= 1,
            w_init_gain= 'linear'
            ))

    def forward(self, x: torch.Tensor):
        return super().forward(x) + x


# Refer: https://github.com/zalandoresearch/pytorch-vq-vae
class Vector_Quantizer(torch.nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_size,
        beta= 0.25,    # commitment_cost
        eps= 1e-5,
        use_ema= True,
        ema_decay= 0.99 # Only for EMA
        ) -> None:
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.beta = beta
        self.eps = eps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        self.embedding = torch.nn.Parameter(torch.empty(num_embeddings, embedding_size))    # [Emb_n, Dim]
        torch.nn.init.xavier_uniform_(self.embedding)

        if use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.ema_weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_size))
            torch.nn.init.normal_(self.ema_weight)

    def forward(self, x):
        # x: [Batch, Ch, H, W]
        batch, channels, height, width = x.size()
        
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, channels)    # [Batch*H*W, Ch]

        # distances = torch.cdist(
        #     x_flat.unsqueeze(1),    # [Batch*H*W, 1, Ch]
        #     self.embedding.unsqueeze(0).expand(x_flat.size(0), -1, -1)    # [Batch*H*W, Emb_n, Ch]
        #     ).squeeze(1)    # [Batch*H*W, Emb_n]
        distances = \
            torch.sum(x_flat**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding ** 2, dim=1) + \
            -2 * torch.matmul(x_flat, self.embedding.T)

        encodings = torch.nn.functional.one_hot(
            distances.argmin(dim= 1),
            num_classes= self.num_embeddings
            ).float()   # [Batch*H*W, Emb_n]
        
        quantized = (encodings @ self.embedding).view(batch, height, width, channels).permute(0, 3, 1, 2)

        if self.training and self.use_ema:
            self.ema_cluster_size = \
                self.ema_decay * self.ema_cluster_size + \
                (1 - self.ema_decay) * torch.sum(encodings, dim= 0)
            n = self.ema_cluster_size.sum()
            self.ema_cluster_size = (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n   # TO SAFE?
            
            delta = encodings.T @ x_flat # [Emb_n, Ch]
            self.ema_weight.data.copy_(self.ema_decay * self.ema_weight + (1 - self.ema_decay) * delta)
            self.embedding.data.copy_(self.ema_weight / self.ema_cluster_size.unsqueeze(1))

        loss = self.Calc_Loss(x, quantized) # Before,

        quantized = x + (quantized - x).detach()    # [Batch*H*W, Ch]
        probabilities = encodings.mean(dim= 0)  # [Emb_n]
        perplexity = torch.exp(-torch.sum(probabilities * torch.log(probabilities + self.eps)))

        return quantized, loss, perplexity

    def Calc_Loss(self, x, quantized):
        e_latent_losses = self.beta * torch.nn.functional.mse_loss(quantized.detach(), x)
        if self.use_ema:
            return e_latent_losses

        q_latent_losses = torch.nn.functional.mse_loss(quantized, x.detach())
        return q_latent_losses + e_latent_losses


class Conv2d(torch.nn.Conv2d):
    def __init__(self, w_init_gain= 'relu', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def __init__(self, w_init_gain= 'relu', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)