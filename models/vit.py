# Path: models/vit.py
# Homemade Vision Transformer

import torch
import torch.nn as nn

class ViT(nn.Module):  # dim=1024, depth=6, heads=8, mlp_dim=2048
    def __init__(self, image_size=224, patch_size=16, num_classes=2, dim=256, depth=2, heads=2, mlp_dim=512):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = nn.Transformer(dim, heads, depth, mlp_dim)
        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img):
        p = self.patch_size
        # B = batch size, C = channels, H = height, W = width
        B, C, H, W = img.shape
        # assert sq image
        assert H == W, "Image dimensions must be square"
        # assert divisible by patch size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by the patch size."
        # number of patches
        n = H // p
        # split into patches
        img = img.reshape(B, C, n, p, n, p).transpose(3, 4)
        img = img.reshape(B, C, n * n, p * p)
        # flatten patches
        img = img.transpose(2, 1).reshape(B, n * n, C * p * p)
        # add patch embeddings
        img = self.to_patch_embedding(img)
        # add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        img = torch.cat((cls_tokens, img), dim=1)
        # add position embeddings
        img += self.pos_embedding[:, :(n * n + 1)]
        img = self.dropout(img)
        # transformer
        img = self.transformer(img)
        # extract class token
        img = self.to_cls_token(img[:, 0])
        # mlp head
        return self.mlp_head(img)
