# python3 test.py --gpu_ids 0 --test_dataroot ./samples --ckpt ./checkpoints/8000.pth
import torch
from .dataset import ImageDatasetTest
from .networks import RegressionNetwork
import argparse
import os
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
# Grad-CAM
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./output/your_root")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument("--test_dataroot", default="./samples")
    parser.add_argument('--ckpt', type=str, help='save checkpoint infos',default="./checkpoints/8000.pth")
    return opt

def visualize(test_iter, model):
    image, label = test_iter.next()

def test(model, test_loader, output_dir):
    model.eval()
    model.cuda()
    # Check folder
    if not os.path.exists("./output"):
        os.mkdir("./output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, "saliency"))
        os.mkdir(os.path.join(output_dir, "gradcam"))
        os.mkdir(os.path.join(output_dir, "gradcam-rgb"))
    # Grad-CAM
    target_layers = [model.resnet.layer4[-1]]
    # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # targets = [ClassifierOutputTarget(0)]
    samples = []
    for i, (image, path) in enumerate(test_loader):
        image = image.cuda()
        image.requires_grad_()
        output = model(image)
        print(output)
        with open(os.path.join(output_dir, "output.txt"), "a") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(output[j].item()) + "\n")
                samples.append("name : " + path[j])
                samples.append("value : " + str(output[j].item()))
        output_sum = torch.sum(output)
        output_sum.backward()
        # Visualize the saliency map
        for j in range(len(output)):
            saliency = torch.sum(image.grad[j].abs(), dim=0)
            saliency = F.relu(saliency)
            saliency = saliency/saliency.max()
            # Save the saliency map
            saliency = saliency.cpu().detach().numpy() * 255
            saliency = Image.fromarray(saliency)
            # Convert to L
            saliency = saliency.convert('L')

            # saliency = saliency * image[j]
            # saliency = torchvision.transforms.ToPILImage()((saliency * 0.5 + 0.5) * 255)
            saliency.save(os.path.join(output_dir, "saliency", os.path.basename(path[j])))
        # Visualize the Grad-CAM
        # grayscale_cam = cam(input_tensor=image, targets=targets, aug_smooth=True)
        # print("Min/max/mean of grayscale_cam", grayscale_cam.min(), grayscale_cam.max(), grayscale_cam.mean())
        # for j in range(len(output)):
        #     img = (image[j]*0.5+0.5).detach().cpu().numpy().transpose(1,2,0)
        #     visualization = show_cam_on_image(img, grayscale_cam[j], use_rgb=True)
        #     # Save the Grad-CAM
        #     visualization = Image.fromarray(visualization)
        #     visualization.save(os.path.join(opt.output_dir, "gradcam-rgb", os.path.basename(path[j]))) 
        #     visualization = Image.fromarray(grayscale_cam[j]*255).convert('L')
        #     visualization.save(os.path.join(opt.output_dir, "gradcam", os.path.basename(path[j])))
    
        print(i)
        return samples

if __name__ == '__main__':
    opt = get_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    # Define model
    model = RegressionNetwork()
    # Load checkpoint
    model.load_state_dict(torch.load(opt.ckpt))
    # Define dataloader
    test_dataset = ImageDatasetTest(data_dir=opt.test_dataroot)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
    # Train the model
    test(model, test_loader, opt)
