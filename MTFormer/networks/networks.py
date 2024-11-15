import torch, torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.shape[0], -1)

class CLS_head(nn.Module):
    def __init__(self, in_channels, out_channels, pooling='avg', dropout=0.2): 
        super(CLS_head, self).__init__()

        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, input):
        linear = self.linear(input)
        return linear

class MTL2d_DeepSVDD(nn.Module):
    def __init__(self, num_class, num_head = 3, att_layer = [0, 1, 2], num_layer = 3, input_size = 64, in_channels=1):
        super(MTL2d_DeepSVDD, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.attn1 = LinearAttention(dim = 16, heads = num_head, dim_head=16//num_head, attn_drop=0.1, proj_drop=0.1, reduce_size=input_size//2, projection='interp', rel_pos=True)

        if input_size < 32:
            num_layer = min(2, num_layer)
        if num_layer>=2:
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.attn2 = LinearAttention(dim = 32, heads = num_head, dim_head=32//num_head, attn_drop=0.1, proj_drop=0.1, reduce_size=input_size//4, projection='interp', rel_pos=True)
            
        if num_layer>=3:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.attn3 = LinearAttention(dim = 64, heads = num_head, dim_head=64//num_head, attn_drop=0.1, proj_drop=0.1, reduce_size=input_size//8, projection='interp', rel_pos=True)
        
        if num_layer>=4:
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.attn4 = LinearAttention(dim = 128, heads = num_head, dim_head=128//num_head, attn_drop=0.1, proj_drop=0.1, reduce_size=input_size//16, projection='interp', rel_pos=True)
        
        if num_layer>=5:
            self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
            self.attn5 = LinearAttention(dim = 256, heads = num_head, dim_head=256//num_head, attn_drop=0.1, proj_drop=0.1, reduce_size=input_size//32, projection='interp', rel_pos=True)

        in_chan = 8*(2**num_layer)
        self.cls = CLS_head(in_channels=in_chan, out_channels=num_class, pooling='avg', dropout=0.5) 
    

        self.num_layer = num_layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()
        self.att_layer = att_layer
    def forward(self, x):
        z = F.relu(self.conv1(x))
        
        if 0 in self.att_layer:
            out, q_k_attn = self.attn1(z)
            z = out + z 

        if self.num_layer>=2:
            z = F.relu(self.conv2(z))

            if 1 in self.att_layer:
                out, q_k_attn = self.attn2(z)
                z = out + z 

        if self.num_layer>=3:
            z = F.relu(self.conv3(z))

            if 2 in self.att_layer:
                out, q_k_attn = self.attn3(z)
                z = out + z 

        if self.num_layer>=4:
            z = F.relu(self.conv4(z))

            if 3 in self.att_layer:
                out, q_k_attn = self.attn4(z)
                z = out + z 

        if self.num_layer>=5:
            z = F.relu(self.conv5(z))

            if 4 in self.att_layer:
                out, q_k_attn = self.attn5(z)
                z = out + z 

        z = self.avg_pool(z)        # 2d feature to 1d feature
        z = self.flatten(z)         # 2d feature to 1d feature
        cls_output = self.cls(z)    # MTL classification 
        return z, cls_output


########################################################################
# Transformer components

from einops import rearrange
class LinearAttention(nn.Module):
    
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=32, projection='interp', rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads # 3 * 21
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head #21?
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos
        
        # depthwise conv is slightly better than conv1x1
        self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True) # Simple QKV
        
        self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
       
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
    def forward(self, x):

        B, C, H, W = x.shape 
        
        #B, inner_dim, H, W
        qkv = self.to_qkv(x)  
        q, k, v = qkv.chunk(3, dim=1) 

        if self.projection == 'interp' and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v)) 

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=H, w=W) 
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v)) 

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        
        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W) 
            q_k_attn += relative_position_bias

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head, heads=self.heads)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn
    

class RelativePositionBias(nn.Module):
    # input-independent relative position attention
    # As the number of parameters is smaller, so use 2D here
    # Borrowed some code from SwinTransformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        self.relative_position_bias_table = nn.Parameter(
                torch.randn((2*h-1) * (2*w-1), num_heads)*0.02) #torch.Size([225, 4])

        coords_h = torch.arange(self.h) #tensor([0, 1, 2, 3, 4, 5, 6, 7])
        coords_w = torch.arange(self.w) #tensor([0, 1, 2, 3, 4, 5, 6, 7])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2, h, w #torch.Size([2, 8, 8])
        coords_flatten = torch.flatten(coords, 1) # 2, hw #torch.Size([2, 64])

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  #torch.Size([2, 64, 64])
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() #torch.Size([64, 64, 2])
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1) # hw, hw
    
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h, self.w, self.h*self.w, -1) #h, w, hw, nH
        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H//self.h, dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W//self.w, dim=1) #HW, hw, nH
        relative_position_bias_expanded = relative_position_bias_expanded.view(H*W, self.h*self.w, self.num_heads).permute(2, 0, 1).contiguous().unsqueeze(0)
        
        return relative_position_bias_expanded

