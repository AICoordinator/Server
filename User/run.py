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
from torchvision.utils import save_image
save_path = 'User/test'
output_dir = 'User/test/'

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


def test(user_email, model,model_mask, save_path, test_loader):
    test_start = time.time()
    # make result list
    delete_images = []
    for i, (image, path, image_big) in enumerate(test_loader):
        image = image
        image_big = image_big
        image.requires_grad_()
        output, feat = model(image, return_feat=True)
        """ with open(os.path.join("User/output/output.txt"), "w") as f:
            for j in range(len(output)):
                f.write(os.path.basename(path[j]) + " " + str(output[j].item()) + "\n")"""
        segmap = model_mask(image_big)[0]
        segmap = F.interpolate(segmap, size=(image.shape[2], image.shape[3]), mode='nearest')
        segmap = segmap.detach().argmax(1).unsqueeze(1)

        mask_skin = (segmap == 1).type(torch.float32) + (segmap == 2).type(torch.float32) + (segmap == 3).type(torch.float32)
        mask_nose = (segmap == 10).type(torch.float32)
        mask_eyes = (segmap == 4).type(torch.float32) + (segmap == 5).type(torch.float32)
        mask_ears = (segmap == 7).type(torch.float32) + (segmap == 8).type(torch.float32)
        mask_mouth = (segmap == 11).type(torch.float32) + (segmap == 12).type(torch.float32) + (segmap == 13).type(torch.float32)
        mask_hair = (segmap == 17).type(torch.float32)
        mask_others = torch.ones_like(mask_skin) - mask_skin - mask_nose - mask_eyes - mask_ears - mask_mouth - mask_hair

        region_skin = mask_skin * (image * 0.5 + 0.5)
        region_nose = mask_nose * (image * 0.5 + 0.5)
        region_eyes = mask_eyes * (image * 0.5 + 0.5)
        region_ears = mask_ears * (image * 0.5 + 0.5)
        region_mouth = mask_mouth * (image * 0.5 + 0.5)
        region_hair = mask_hair * (image * 0.5 + 0.5)

        save_image(region_skin, os.path.join(output_dir, "skin.jpg"))
        save_image(region_nose, os.path.join(output_dir, "nose.jpg"))
        save_image(region_eyes, os.path.join(output_dir, "eyes.jpg"))
        save_image(region_ears, os.path.join(output_dir, "ears.jpg"))
        save_image(region_mouth, os.path.join(output_dir, "mouth.jpg"))
        save_image(region_hair, os.path.join(output_dir, "hair.jpg"))

        print(f"feat.shape : {feat.shape}, mask_skin.shape : {mask_skin.shape}")
        output_sum = torch.sum(output)
        feat.retain_grad()
        output_sum.backward()

        grayscale_cams = feat
        print(f"grayscale_cams.shape : {grayscale_cams.shape}")
        print(f"mask_skin.shape : {mask_skin.shape}")

        grayscale_cams = F.interpolate(grayscale_cams, size=image.shape[2:], mode='bicubic', align_corners=True)
        print(f"grayscale_cam  s.shape : {grayscale_cams.shape}")

        offset = 20
        score_skin = offset - (grayscale_cams * mask_skin).sum(dim=(1, 2, 3)) / mask_skin.sum(dim=(1, 2, 3))
        score_nose = offset - (grayscale_cams * mask_nose).sum(dim=(1, 2, 3)) / mask_nose.sum(dim=(1, 2, 3))
        score_eyes = offset - (grayscale_cams * mask_eyes).sum(dim=(1, 2, 3)) / mask_eyes.sum(dim=(1, 2, 3))
        score_ears = offset - (grayscale_cams * mask_ears).sum(dim=(1, 2, 3)) / mask_ears.sum(dim=(1, 2, 3))
        score_mouth = offset - (grayscale_cams * mask_mouth).sum(dim=(1, 2, 3)) / mask_mouth.sum(dim=(1, 2, 3))
        score_hair = offset - (grayscale_cams * mask_hair).sum(dim=(1, 2, 3)) / mask_hair.sum(dim=(1, 2, 3))

        # if nan, assign 5
        score_skin[torch.isnan(score_skin)] = offset
        score_nose[torch.isnan(score_nose)] = offset
        score_eyes[torch.isnan(score_eyes)] = offset
        score_ears[torch.isnan(score_ears)] = offset
        score_mouth[torch.isnan(score_mouth)] = offset
        score_hair[torch.isnan(score_hair)] = offset

        std_skin = grayscale_cams[mask_skin.type(torch.bool)].std()
        std_nose = grayscale_cams[mask_nose.type(torch.bool)].std()
        std_eyes = grayscale_cams[mask_eyes.type(torch.bool)].std()
        std_ears = grayscale_cams[mask_ears.type(torch.bool)].std()
        std_mouth = grayscale_cams[mask_mouth.type(torch.bool)].std()
        std_hair = grayscale_cams[mask_hair.type(torch.bool)].std()

        # If std is less than 1e-4, std = 1
        std_skin[std_skin < 1e-4] = 1
        std_nose[std_nose < 1e-4] = 1
        std_eyes[std_eyes < 1e-4] = 1
        std_ears[std_ears < 1e-4] = 1
        std_mouth[std_mouth < 1e-4] = 1
        std_hair[std_hair < 1e-4] = 1

        # print std
        print(f"std_skin : {std_skin}")
        print(f"std_nose : {std_nose}")
        print(f"std_eyes : {std_eyes}")
        print(f"std_ears : {std_ears}")
        print(f"std_mouth : {std_mouth}")
        print(f"std_hair : {std_hair}")

        # Check if grayscale_cams contains nan
        if torch.isnan(grayscale_cams).any():
            print("grayscale_cams contains nan")
            print(grayscale_cams)
            raise NotImplementedError

        score_mean_skin = (offset - score_skin).mean()
        score_mean_nose = (offset - score_nose).mean()
        score_mean_eyes = (offset - score_eyes).mean()
        score_mean_ears = (offset - score_ears).mean()
        score_mean_mouth = (offset - score_mouth).mean()
        score_mean_hair = (offset - score_hair).mean()

        print(f"score_mean_skin: {score_mean_skin}")
        print(f"score_mean_nose: {score_mean_nose}")
        print(f"score_mean_eyes: {score_mean_eyes}")
        print(f"score_mean_ears: {score_mean_ears}")
        print(f"score_mean_mouth: {score_mean_mouth}")
        print(f"score_mean_hair: {score_mean_hair}")

        for j in range(len(output)):
            # Select top/bottom 10% pixels
            grayscale_cam = grayscale_cams[j]
            print(f"grayscale_cam.shape : {grayscale_cam.shape}")

            grayscale_cam -= mask_skin[j] * score_mean_skin
            grayscale_cam -= mask_nose[j] * score_mean_nose
            grayscale_cam -= mask_eyes[j] * score_mean_eyes
            grayscale_cam -= mask_ears[j] * score_mean_ears
            grayscale_cam -= mask_mouth[j] * score_mean_mouth
            grayscale_cam -= mask_hair[j] * score_mean_hair

            # Divide by std
            grayscale_cam[mask_skin[j].type(torch.bool)] = grayscale_cam[mask_skin[j].type(torch.bool)] / std_skin
            grayscale_cam[mask_nose[j].type(torch.bool)] = grayscale_cam[mask_nose[j].type(torch.bool)] / std_nose
            grayscale_cam[mask_eyes[j].type(torch.bool)] = grayscale_cam[mask_eyes[j].type(torch.bool)] / std_eyes
            # grayscale_cam[mask_ears[j].type(torch.bool)] = grayscale_cam[mask_ears[j].type(torch.bool)] / std_ears
            grayscale_cam[mask_mouth[j].type(torch.bool)] = grayscale_cam[mask_mouth[j].type(torch.bool)] / std_mouth
            grayscale_cam[mask_hair[j].type(torch.bool)] = grayscale_cam[mask_hair[j].type(torch.bool)] / std_hair

            print(f"skin: {(grayscale_cam[mask_skin[j].type(torch.bool)]).mean()}")
            print(f"nose: {(grayscale_cam[mask_nose[j].type(torch.bool)]).mean()}")
            print(f"eyes: {(grayscale_cam[mask_eyes[j].type(torch.bool)]).mean()}")
            print(f"ears: {(grayscale_cam[mask_ears[j].type(torch.bool)]).mean()}")
            print(f"mouth: {(grayscale_cam[mask_mouth[j].type(torch.bool)]).mean()}")
            print(f"hair: {(grayscale_cam[mask_hair[j].type(torch.bool)]).mean()}")

            print(f"---------------------------")

            grayscale_cam = normalize(grayscale_cam)
            grayscale_cam[mask_others[j].type(torch.bool)] = 0
            # save_image(grayscale_cam, f"{output_dir}/{j}_map.jpg")
            grayscale_cam = grayscale_cam.cpu().detach().numpy().squeeze()
            img = (image[j]).cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            # Save the Grad-CAM
            visualization = Image.fromarray(visualization)
            print(output[j].item())
            user_image = UserImage()
            user = User.objects.get(email=user_email)
            user_image.owner = user
            bytearr = io.BytesIO()
            visualization.save(bytearr, format="jpeg")
            with open(save_path + os.path.basename(path[j]), "rb") as image_file:
                user_image.originImage = (base64.b64encode(image_file.read()).decode('utf-8'))
                user_image.changedImage = (base64.b64encode(bytearr.getvalue()).decode('utf-8'))
                user_image.title = save_path + os.path.basename(path[j])
                print('a   ' + user_image.title)
                delete_images.append(user_image.title)
                user_image.score = str(round((float(output[j].item()) + user.pvalue) * 20, 1))
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
    return test(user_email, UserConfig.model,UserConfig.model_mask, test_dataroot, test_loader)


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