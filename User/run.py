import base64
import threading
import time
from math import floor

import torch
from .dataset import ImageDatasetTest
import cv2
import os
from .networks import ResNet18
from .test import test
import argparse
from .models import UserImage,User,File
from .utils import normalize
import torch.nn.functional as F
from pytorch_grad_cam.utils.image import show_cam_on_image
from .app import UserConfig
import numpy as np
from PIL import Image
from .models import UserImage
import io
save_path = 'User/test'

def extract_images(user_email, video_path, save_path, num_images=10, interval = 30):
    # Load the video
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    success = True
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    while success:
        success, image = vidcap.read()
        if count % interval == 0:
            print(save_path+f"{count // interval}.jpg")
            cv2.imwrite(os.path.join(save_path, f"{count // interval}.jpg"), image)     # save frame as JPEG file
            if count // interval == (num_images - 1):
                break
        count += 1
    delete_thread = threading.Thread(target=delete_video,args=(user_email,video_path))
    delete_thread.start()


def test(user_email, model,save_path, test_loader):
    test_start = time.time()
    # make result list
    delete_images = []
    for i, (image, path, image_big) in enumerate(test_loader):
        image = image
        image.requires_grad_()
        output, feat = model(image, return_feat=True)
        """ with open(os.path.join("User/output/output.txt"), "w") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(output[j].item()) + "\n")"""

        output_sum = torch.sum(output)
        feat.retain_grad()
        output_sum.backward()
        grayscale_cams = feat
        grayscale_cams = F.interpolate(grayscale_cams, size=image_big.shape[2:], mode='bicubic',align_corners=True).squeeze()
        for j in range(len(output)):
            # Select top/bottom 10% pixels
            grayscale_cam = grayscale_cams[j]
            print("grayscale_cam min max", grayscale_cam.min(), grayscale_cam.max())
            k = grayscale_cam.shape[-1] * grayscale_cam.shape[-2] // 10
            top_k, _ = torch.topk(grayscale_cam.flatten(), k=k)
            bottom_k, _ = torch.topk(-grayscale_cam.flatten(), k=k)
            bottom_k *= -1
            grayscale_cam_top = torch.maximum(top_k.min(), grayscale_cam)
            grayscale_cam_bot = torch.minimum(bottom_k.max(), grayscale_cam)
            grayscale_cam_bot = normalize(grayscale_cam_bot).cpu().detach().numpy()
            grayscale_cam_top = normalize(grayscale_cam_top).cpu().detach().numpy()
            grayscale_cam = normalize(grayscale_cam).cpu().detach().numpy()

            img = (image_big[j]).detach().numpy().transpose(1, 2, 0)
            visualization = show_cam_on_image(img, grayscale_cam_top, use_rgb=True)
            # Save the Grad-CAM
            visualization = Image.fromarray(visualization)
            #visualization.save(os.path.join(output_dir, os.path.basename(path[j])))
            user_image = UserImage()
            user_image.owner = User.objects.get(email=user_email)
            bytearr = io.BytesIO()
            visualization.save(bytearr, format="jpeg")
            with open(save_path + os.path.basename(path[j]), "rb") as image_file:
                user_image.originImage = (base64.b64encode(image_file.read()).decode('utf-8'))
                user_image.changedImage = (base64.b64encode(bytearr.getvalue()).decode('utf-8'))
                user_image.title = save_path+os.path.basename(path[j])
                print('a   ' + user_image.title)
                delete_images.append(user_image.title)
                user_image.score = str(round(float(output[j].item()) *20, 1))
                user_image.save()
    test_end = time.time()
    delete_thread = threading.Thread(target=delete_image, args=(delete_images,))
    delete_thread.start()
    print(f"function time : {test_end - test_start: .5f} sec")


def run_test(user_email,unique_key, num_images, interval, batch_size):
    """
    video_path: path to the video
    test_dataroot: path where the extracted images are saved
    num_images: number of images to extract
    interval: interval between two consecutive images
    ckpt: path to the checkpoint
    output_dir: path where the output files are saved
    """
    start_extract = time.time()

    video_path = getVedioPath(user_email)
    test_dataroot = get_test_data_root(user_email,unique_key)
    extract_images(user_email, video_path, test_dataroot, num_images, interval)
    end_extract = time.time()
    print(f"extract image time : {end_extract - start_extract: .5f} sec")

    # Define dataloader
    test_dataset = ImageDatasetTest(data_dir=test_dataroot)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # Inference
    return test(user_email, UserConfig.model,test_dataroot, test_loader)


def getVedioPath(user_email):
    video_path = user_email.split('@')[0]
    video_path += user_email.split('@')[1]
    video_path = video_path.split('.')[0]
    print(video_path + '.mp4')
    return 'User/media/video/'+video_path+'.mp4'


def get_test_data_root(user_email,unique_key):
    return 'User/pimages/'+user_email+'/'+unique_key+'/'


def delete_video(user_email, video_path):
    file = File.objects.filter(owner=user_email)
    file.delete()
    if os.path.isfile(video_path):
        os.remove(video_path)


def delete_image(image_path):
    for image in image_path:
        if os.path.isfile(image):
            os.remove(image)