import argparse
import os
import random
import glob

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", type=str, default='eval_configs/minigpt4.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--image-folder", type=str, required=True, help="path to the input image folder")
    parser.add_argument("--beam-search-numbers", type=int, default=1, help="beam search numbers")
    options = parser.parse_args()
    return options


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def describe_image(image_path, chat, chat_state, img, num_beams=1, temperature=1.0):
    chat_state = CONV_VISION.copy()
    img_list = []

    gr_img = Image.open(image_path)
    llm_message = chat.upload_img(gr_img, chat_state, img_list)

    chat.ask("Describe this image.", chat_state)
    generated_caption = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=300, num_beams=num_beams, temperature=temperature, max_length=2000)[0]

    return generated_caption


if __name__ == '__main__':
    args = parse_args()

    cfg = Config(args)

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')

    vis_processor_cfg = cfg.datasets_cfg.cc_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor)

    chat_state = CONV_VISION.copy()
    img_list = []

    image_folder = args.image_folder
    num_beams = args.beam_search_numbers
    temperature = 1.0  # default temperature

    image_extensions = ['jpg', 'jpeg', 'png', 'bmp']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, f'*.{ext}')))
        image_paths.extend(glob.glob(os.path.join(image_folder, f'*.{ext.upper()}')))
    
    if not os.path.exists("mycaptions"):
        os.makedirs("mycaptions")

    for image_path in image_paths:
        caption = describe_image(image_path, chat, chat_state, img_list, num_beams, temperature)

        with open("mycaptions/{}_caption.txt".format(os.path.splitext(os.path.basename(image_path))[0]), "w") as f:
            f.write(caption)
        
        print(f"Caption for {os.path.basename(image_path)} saved in 'mycaptions' folder")