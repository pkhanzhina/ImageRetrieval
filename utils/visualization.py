from torchvision import transforms


def prepare_img(tensor):
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return inv_normalize(tensor).squeeze(0).permute(1, 2, 0).numpy() * 255
