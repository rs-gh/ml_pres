import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


ATTENTION_DEBUG = False

device = torch.device("cuda:0") if torch.cuda_is_available() else torch.device("cpu")


def attention_function(query, key, value, mask=None):
    d_k = query.shape[-1] * 1.0
    # (batch_size, num_heads, seq_len, seq_len)
    logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        if ATTENTION_DEBUG:
            print(f"logits shape: {logits.shape}")
            print(f"mask shape: {mask.shape}")
        # When logit is set to -inf, softmax(logit) = 0, which is what we want
        logits = logits.masked_fill_(mask == 0, -float("inf"))

    attention = nn.Softmax(dim=-1)(logits)
    values = torch.matmul(attention, value)
    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_query,
        d_key,
        d_value,
        d_model,
        num_heads,
        attention_function=attention_function,
    ):
        assert d_model % num_heads == 0, "d_model must be 0 modulo num_heads."
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.attention_function = attention_function
        self.query_linear = nn.Linear(d_query, d_model)
        self.key_linear = nn.Linear(d_key, d_model)
        self.value_linear = nn.Linear(d_value, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def get_heads(self, x):
        batch_len, seq_len = x.shape[0], x.shape[1]
        x = x.reshape(
            batch_len, seq_len, self.num_heads, self.d_model // self.num_heads
        )
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, query, key, value, mask=None, return_attention=False):
        query_batch_len, query_seq_len = query.shape[0], query.shape[1]
        query = self.get_heads(self.query_linear(query))
        key = self.get_heads(self.key_linear(key))
        value = self.get_heads(self.value_linear(value))

        output, attention = self.attention_function(query, key, value, mask)
        output = output.permute(0, 2, 1, 3).reshape(
            (query_batch_len, query_seq_len, self.d_model)
        )
        output = self.output_linear(output)

        if not return_attention:
            return output
        return output, attention


class FeedForward(nn.Module):
    def __init__(self, d_input, d_feedforward):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_input, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_input),
        )

    def forward(self, x):
        return self.feed_forward(x)


class Encoder(nn.Module):
    def __init__(self, d_input, num_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_input, d_input, d_input, d_input, num_heads
        )
        self.layer_norm_1 = nn.LayerNorm(d_input)
        self.feed_forward = FeedForward(d_input, d_input * 4)
        self.layer_norm_2 = nn.LayerNorm(d_input)

    def forward(self, x, mask=None):
        # Pre-LN layer
        ln_x = self.layer_norm_1(x)
        x = x + self.self_attention(ln_x, ln_x, ln_x, mask=mask)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, num_encoders=6, **encoder_args):
        super().__init__()
        self.encoders = nn.ModuleList(
            [Encoder(**encoder_args) for _ in range(num_encoders)]
        )

    def forward(self, x, mask=None):
        for encoder in self.encoders:
            x = encoder(x, mask)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for encoder in self.encoders:
            _, attention_map = encoder.self_attention(
                x, x, x, mask, return_attention=True
            )
            attention_maps.append(attention_map)
            x = encoder(x)
        return attention_maps


class Decoder(nn.Module):
    def __init__(self, d_input, d_encoder_memory, num_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_input, d_input, d_input, d_input, num_heads
        )
        self.layer_norm_1 = nn.LayerNorm(d_input)
        self.encoder_attention = MultiHeadAttention(
            d_input, d_encoder_memory, d_encoder_memory, d_input, num_heads
        )
        self.layer_norm_2 = nn.LayerNorm(d_input)
        self.feed_forward = FeedForward(d_input, d_input * 3)
        self.layer_norm_3 = nn.LayerNorm(d_input)

    def forward(self, x, encoder_memory, mask=None):
        x = self.layer_norm_1(x + self.self_attention(x, x, x, mask))
        x = self.layer_norm_2(
            x
            + self.encoder_attention(query=x, key=encoder_memory, value=encoder_memory)
        )
        x = self.layer_norm_3(x + self.feed_forward(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, num_decoders=6, **decoder_args):
        super().__init__()
        self.decoders = nn.ModuleList(
            [Decoder(**decoder_args) for _ in range(num_decoders)]
        )

    def forward(self, x, encoder_memory, mask=None):
        for decoder in self.decoders:
            x = decoder(x, encoder_memory, mask)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, encoder_memory, mask=None):
        self_attention_maps, encoder_attention_maps = [], []

        for decoder in self.decoders:
            outputs, self_attention_map = decoder.self_attention(
                x, x, x, mask, return_attention=True
            )
            self_attention_maps.append(self_attention_map)

            _, encoder_map = decoder.encoder_attention(
                query=outputs,
                key=encoder_memory,
                value=encoder_memory,
                mask=mask,
                return_attention=True,
            )
            encoder_attention_maps.append(encoder_map)
            x = decoder(x)


class LinearProjection(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.projection = nn.Linear(d_input, d_output)

    def forward(self, x):
        x = self.projection(x)
        return x


def convert_images_to_flat_patches(x, patch_size, flatten_channels=True):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # (b, h/p, w/p, c, p_h, p_w)
    x = x.flatten(1, 2)  # (b, h/p * w/p, c, p_h, p_w)
    if flatten_channels:
        x = x.flatten(2, 4)  # (b, h/p * w/p, c * p_h * p_w)
    return x


class VisionTransfomer(nn.Module):
    def __init__(
        self,
        d_model,
        num_encoders,
        num_heads,
        num_channels,
        patch_size,
        max_patches,
        num_classes,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.linear_projection = LinearProjection(
            num_channels * patch_size**2, d_model
        )
        self.encoders = EncoderBlock(
            d_input=d_model, num_encoders=num_encoders, num_heads=num_heads
        )
        self.mlp_head = (
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))
            if num_classes > 0
            else nn.Identity()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + max_patches, d_model))

    def forward(self, x):
        x = convert_images_to_flat_patches(x, self.patch_size)
        batch_size, num_patches, _ = x.shape
        x = self.linear_projection(x)  # batch, num_patches, d_model
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        # add a class token to every picture (that is now flat), one for each pic
        x = torch.cat([cls_token, x], dim=1)
        # add learnable positional embedding, one for the class token + per patch
        x = x + self.pos_embedding[:, 1 + num_patches]
        x = self.encoders(x)
        # classify the class token
        return self.mlp_head(x[:, 0])


class ViT(pl.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransfomer(**model_kwargs)

    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, batch, mode="train"):
        x, true = batch
        predicted = self.forward(x)
        loss = F.cross_entropy(predicted, true)
        acc = (predicted.argmax(dim=-1) == true).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]
