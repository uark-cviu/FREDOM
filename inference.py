import torch
import torchvision
import torchvision.transforms as T
import argparse
import numpy as np
from PIL import Image

from models.deeplabv2 import DeepLabV2_ResNet101

COLORS = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 0)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type = str)
    parser.add_argument("--input_path", type = str)
    parser.add_argument("--output_path", type = str)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    model = DeepLabV2_ResNet101(num_classes = 19)
    checkpoint = torch.load(args.checkpoint)
    msg = model.load_state_dict(checkpoint)
    print(f"Loading model from {args.checkpoint}: {msg}")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    image = Image.open(args.input_path)
    image = image.convert("RGB")
    (width, height) = (image.width // 2, image.height // 2) # Save memory of GPU
    image = image.resize((width, height))
    image = transform(image)
    image = image.unsqueeze(0)

    if torch.cuda.is_available():
        image = image.cuda()
        model.cuda()

    _, output = model(image)
    output = output[0].cpu().detach().numpy()
    output = np.argmax(output, axis = 0)

    palette = []
    for color in COLORS:
        palette += color
    zero_pad = 256*3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    mask = Image.fromarray(output.astype(np.uint8)).convert('P')
    mask.putpalette(palette)
    mask = np.asarray(mask.convert("RGB"))

    image = Image.open(args.input_path)
    image = image.convert("RGB")
    image = image.resize((width, height))
    image = np.asarray(image)

    image = image * args.alpha + mask * (1 - args.alpha)
    image = image.astype(np.uint8)
    image = Image.fromarray(image).convert("RGB")
    image.save(args.output_path)



