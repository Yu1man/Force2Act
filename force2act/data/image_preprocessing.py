import torch

def preprocess_images(images: torch.Tensor) -> torch.Tensor:
    # [B, N_cam, T_obs, 3, H, W]

    ndim = images.ndim

    leading_dims = ndim - 3

    if images.ndim < 3:
        raise ValueError(f"Images dim cannot be smaller than 3")

    if images.shape[leading_dims] != 3:
        raise ValueError(f"Images channels has to be 3 (RGB)")

    if images.dtype != torch.uint8:
        raise ValueError(f"Images dtype ahs to be uint8")


    images = images.float()

    images = images / 255.0

    stat_shape = [1] * leading_dims + [3, 1, 1]

    mean = images.new_tensor([0.485, 0.456, 0.406])
    std = images.new_tensor([0.229, 0.224, 0.225])

    mean = mean.reshape(stat_shape)
    std = std.reshape(stat_shape)

    images = (images-mean) / std

    return images



if __name__ == "__main__":

    images = torch.randint(
        0, 256,
        (2, 3, 3, 64, 64),
        dtype=torch.uint8
    )

    processed = preprocess_images(images)

    print(processed.shape)
    print(processed.dtype)

    images = torch.full(
        (1, 1, 1, 3, 2, 2),
        255,
        dtype=torch.uint8
    )

    processed = preprocess_images(images)

    print(processed[0, 0, 0, :, 0, 0])