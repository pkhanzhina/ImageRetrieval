import torch.nn as nn
from models.block._resnet import resnet50


class Resnet50(nn.Module):
    def __init__(self, embed_size, pretrained=True, with_norm=False):
        super().__init__()
        self.embed_size = embed_size
        self.with_norm = with_norm
        self.main_model = resnet50(pretrained=pretrained, num_classes=embed_size)

    def forward(self, input):
        output = self.main_model(input)
        if self.with_norm:
            output = nn.functional.normalize(output, p=2.0, dim=-1)
        return output