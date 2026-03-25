
import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import RgbTestDataset
from fusion_model import get_fusion_model
import args

def test(test_vi_path, test_ir_path, fusion_results_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(fusion_results_path, exist_ok=True)

    model = get_fusion_model().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False))

    test_dataset = RgbTestDataset(test_vi_path, test_ir_path, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for idx, (vi_images, ir_images,  vi_image_name, ir_image_name) in enumerate(test_loader):

            vi_images_np = vi_images.numpy().squeeze().transpose(1, 2, 0)
            vi_images_ycrcb = cv2.cvtColor(vi_images_np, cv2.COLOR_RGB2YCrCb)
            vi_images_y = vi_images_ycrcb[:, :, 0:1]


            vi_images_y = torch.from_numpy(vi_images_y).permute(2, 0, 1).unsqueeze(0).to(device)
            ir_images, vi_images_y = ir_images.to(device), vi_images_y.to(device)

            ir_images = ir_images.tile(1,2,1,1)
            vi_images_y = vi_images_y.tile(1,2,1,1)

            fusion_images = model(ir_images, vi_images_y)


            fusion_image_np = fusion_images.squeeze(0).squeeze(0).cpu().numpy()
            vi_images_ycrcb[:, :, 0] = fusion_image_np
            fusion_image_rgb = cv2.cvtColor(vi_images_ycrcb, cv2.COLOR_YCrCb2RGB)

            fusion_image_rgb = fusion_image_rgb * 255

            cv2.imwrite(os.path.join(fusion_results_path, ir_image_name[0]), fusion_image_rgb)


if __name__ == '__main__':

    path_vi = f"../data/images/vi"
    path_ir = f"../data/images/ir"
    path_op = f"../data/images/fusion"

    output_path = f"{path_op}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    IR_PATH = path_ir
    VI_RGB_PATH = path_vi
    SAVE_PATH = output_path

    test_vi_path = VI_RGB_PATH
    test_ir_path = IR_PATH
    fusion_results_path = SAVE_PATH

    model_path = os.path.join(args.model_save_path, 'best_model.pth')
    test(test_vi_path, test_ir_path, fusion_results_path, model_path)
