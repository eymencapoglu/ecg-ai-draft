
import torch
import torch.nn as nn
import math

# Combined helper
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, content_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, content_channels, kernel_size=1),
            Swish(),
            nn.Conv1d(content_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        expanded_channels = in_channels * expand_ratio
        
        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.append(nn.Conv1d(in_channels, expanded_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(expanded_channels))
            layers.append(Swish())
            
        # Depthwise
        layers.append(nn.Conv1d(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm1d(expanded_channels))
        layers.append(Swish())
        
        # SE
        if se_ratio:
            se_inner = max(1, int(in_channels * se_ratio))
            layers.append(SEBlock(expanded_channels, se_inner))
            
        # Output
        layers.append(nn.Conv1d(expanded_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))
        
        self.block = nn.Sequential(*layers)
        self.drop_rate = drop_connect_rate

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            if self.drop_rate > 0 and self.training:
                 # Drop connect
                 keep_prob = 1 - self.drop_rate
                 mask = torch.rand((out.size(0), 1, 1), device=out.device) < keep_prob
                 out = out * mask.float() / keep_prob # Scale to maintain mean
            return x + out
        return out

class EfficientNet1D(nn.Module):
    def __init__(self, num_classes=3, in_channels=12):
        super().__init__()
        # B0 config adapted for 1D
        # width_mult=1.0, depth_mult=1.0
        # (expand, k, s, in, out, repeats)
        settings = [
            [1, 3, 1, 32, 16, 1],
            [6, 3, 2, 16, 24, 2],
            [6, 5, 2, 24, 40, 2],
            [6, 3, 2, 40, 80, 3],
            [6, 5, 1, 80, 112, 3],
            [6, 5, 2, 112, 192, 4],
            [6, 3, 1, 192, 320, 1],
        ]
        
        out_channels = 32
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            Swish()
        )
        
        blocks = []
        for t, k, s, i, o, r in settings:
            # We treat the first block of the group as the one with stride. Others stride 1.
            # But the settings usually imply the first one does the stride.
            # Input channels calculation: we need to update 'out_channels' tracking.
            # The 'i' in settings is the expected input.
            if i != out_channels:
                # This happens if we change width multiplier. here it matches.
                pass
            
            for rep in range(r):
                stride = s if rep == 0 else 1
                input_ch = out_channels if rep == 0 else o # First block takes 'out_channels', subsequent take 'o'
                drop_rate = 0.2 * len(blocks) / 16.0 # linearly increasing drop probability? simplistic
                blocks.append(MBConvBlock(input_ch, o, t, k, stride, drop_connect_rate=drop_rate))
                out_channels = o
                
        self.blocks = nn.Sequential(*blocks)
        
        last_channels = 1280
        self.head = nn.Sequential(
            nn.Conv1d(out_channels, last_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(last_channels),
            Swish(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2), # Standard B0 dropout
            nn.Linear(last_channels, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

if __name__ == "__main__":
    model = EfficientNet1D()
    x = torch.randn(2, 12, 5000)
    print(model(x).shape)
