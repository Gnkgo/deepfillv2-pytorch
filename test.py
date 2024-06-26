import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as T
from glob import glob

parser = argparse.ArgumentParser(description='Batch inpainting')
parser.add_argument("--images_dir", type=str, default="examples/inpaint/images", help="directory with image files")
parser.add_argument("--masks_dir", type=str, default="examples/inpaint/masks", help="directory with mask files")
parser.add_argument("--output_dir", type=str, default="examples/inpaint/outputs", help="directory to save output files")
parser.add_argument("--checkpoint", type=str, default="pretrained/states_tf_places2.pth", help="path to the checkpoint file")

def main():
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    generator_state_dict = torch.load(args.checkpoint)['G']

    if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
        from model.networks import Generator
    else:
        from model.networks_tf import Generator  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up network
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    generator.load_state_dict(generator_state_dict, strict=True)

    # Get list of image and mask files
    image_files = sorted(glob(os.path.join(args.images_dir, '*.png')))
    mask_files = sorted(glob(os.path.join(args.masks_dir, '*.png')))

    for image_path, mask_path in zip(image_files, mask_files):
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Prepare input
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)

        _, h, w = image.shape
        grid = 8

        image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

        image = (image * 2 - 1.).to(device)  # map image values to [-1, 1] range
        mask = (mask > 0.5).to(dtype=torch.float32, device=device)  # 1.: masked 0.: unmasked

        image_masked = image * (1. - mask)  # mask image
        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)  # concatenate channels

        with torch.inference_mode():
            _, x_stage2 = generator(x, mask)

        # Complete image
        image_inpainted = image * (1. - mask) + x_stage2 * mask

        # Save inpainted image
        img_out = ((image_inpainted[0].permute(1, 2, 0) + 1) * 127.5)
        img_out = img_out.to(device='cpu', dtype=torch.uint8)
        img_out = Image.fromarray(img_out.numpy())
        output_path = os.path.join(args.output_dir, os.path.basename(image_path))
        img_out.save(output_path)

        print(f"Saved output file at: {output_path}")

if __name__ == '__main__':
    main()
