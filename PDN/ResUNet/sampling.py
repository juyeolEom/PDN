import torch
from einops import rearrange

def generate_mask_pair(img, generator=None, st_mixing=False):
    # ST mixing draws independent spatial choices for each time bin.
    n, c, t, h, w = img.shape
    h2, w2 = h // 2, w // 2

    idx_pair = torch.tensor([
        [0, 1, 2], [0, 2, 1],
        [1, 0, 3], [1, 3, 0],
        [2, 0, 3], [2, 3, 0],
        [3, 2, 1], [3, 1, 2]],
        dtype=torch.int64,
        device=img.device
    )

    rd_idx = torch.randint(
        0, 8,
        size=(n, t if st_mixing else 1, h2, w2),
        device=img.device,
        generator=generator
    )
    pair = idx_pair[rd_idx]  # n, time choices, h2, w2, 3

    mask1 = torch.zeros((n, t, h2, w2, 4), dtype=torch.bool, device=img.device)
    mask2 = torch.zeros_like(mask1)
    mask3 = torch.zeros_like(mask1)

    pair = pair.expand(n, t, h2, w2, 3)

    mask1.scatter_(-1, pair[..., 0:1], True)
    mask2.scatter_(-1, pair[..., 1:2], True)
    mask3.scatter_(-1, pair[..., 2:3], True)

    return mask1.reshape(-1), mask2.reshape(-1), mask3.reshape(-1)

def generate_subimages(img, mask):
    n, c, t, h, w = img.shape
    img = rearrange(img, 'b c s h w -> (b s) c h w')
    subimage = torch.zeros(n*t,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n*t, h // 2, w // 2, c).permute(0, 3, 1, 2)

    subimage = rearrange(subimage, '(n t) c h w -> n c t h w', n=n, t=t)
    return subimage

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)