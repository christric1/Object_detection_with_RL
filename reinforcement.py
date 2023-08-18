import numpy as np

import torch
import torchvision.transforms.functional as TF
from torch import Tensor


def transform_action(action, x, y):
    '''
        Map actions to a specified range
    '''
    min_value = -1
    max_value = 1
    transformed_array = (action - min_value) * (y - x) / (max_value - min_value) + x
    return transformed_array

def modify_image(image: Tensor, brightness_factor: float, saturation_factor: float, contrast_factor: float, sharpness_factor: float):
    """
        Adjusting the contrast, saturation, brightness and sharpness
    """
    # Adjust saturation & brightness & contrast & sharpness
    bright_img = TF.adjust_brightness(image, brightness_factor)
    saturation_img = TF.adjust_saturation(bright_img, saturation_factor)
    contrast_img = TF.adjust_contrast(saturation_img, contrast_factor)
    sharpness_img = TF.adjust_sharpness(contrast_img, sharpness_factor)
    
    return sharpness_img

def distortion_image(image: Tensor, max_kernel_size=5):
    '''
        Apply random blurring
    '''
    # Random blurring
    kernel_size = torch.randint(1, max_kernel_size+1, size=(1,)).item()
    kernel_size += kernel_size % 2 - 1  # Ensure odd kernel size
    image = TF.gaussian_blur(image, kernel_size)

    return image

def get_score(precison, recall):
    return 0.5 * precison + 0.5 * recall

def get_reward(RL_score, Origin_score):
    def tanh(x, scale=1.0):
        # return scale * (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return (np.exp(scale*x) - np.exp(scale*(-x))) / (np.exp(scale*x) + np.exp(scale*(-x)))

    scale = 4.0
    score = RL_score - Origin_score
    eta = tanh(score, scale)

    return eta