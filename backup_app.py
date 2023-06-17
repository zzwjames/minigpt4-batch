import argparse
import csv
import os
import random
import glob
import time

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

import cv2
from tqdm import tqdm
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
from pathlib import Path
from copy import deepcopy
import keras


#parsing the arguments 

def parse_args():
    parser = argparse.ArgumentParser(description="Combined Demo")
    parser.add_argument("--cfg-path", type=str, default='eval_configs/minigpt4.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    parser.add_argument("--image-folder", type=str, required=True, help="Path to the input image folder.")
    parser.add_argument("--model", type=str, default='llama', help="Model to be used for generation. Options: 'llama' (default), 'llama7b'")
    parser.add_argument("--beam-search-numbers", type=int, default=1, help="beam search numbers")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_WD14_TAGGER_REPO, help="Hugging Face model repository ID.")
    parser.add_argument("--force-download", action="store_true", help="Force download the model.")
    parser.add_argument("--general-threshold", type=float, default=0.5, help="Threshold for general tags.")
    parser.add_argument("--character-threshold", type=float, default=0.5, help="Threshold for character tags.")
    parser.add_argument("--remove-underscore", action="store_true", help="Remove underscores from captions.")
    parser.add_argument("--undesired-tags", type=str, default="", help="Comma separated list of undesired tags.")
    args = parser.parse_args()
    return args


# these are functions taken from minigpt4 app.py
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


#these are functions taken from wd_tags.py 

IMAGE_SIZE = 448
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp",".webp"]

# wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2/ wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2
DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]

def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # Remove duplicates
    image_paths.sort()
    return image_paths


def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)
    return image


class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess_image(image)
            tensor = torch.tensor(image)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (tensor, img_path)


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch

def run_batch(images, model, args):
    # define the tags
    with open(os.path.join(args.model_dir, CSV_FILE), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        header = l[0]  # tag_id, name, category, count
        rows = l[1:]
        assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"unexpected csv format: {header}"
        general_tags = [row[1] for row in rows[1:] if row[2] == "0"]
        character_tags = [row[1] for row in rows[1:] if row[2] == "4"]

    undesired_tags = set(args.undesired_tags.split(","))

    # Process images to generate captions
    probs = model(np.array(images), training=False)
    captions = []
    for batch_probs in probs.numpy():
        tag_text = ""
        for i, p in enumerate(batch_probs[4:]):
            if i < len(general_tags) and p >= args.general_threshold:
                tag_name = general_tags[i]
                tag_name = tag_name if not args.remove_underscore or len(tag_name) <= 3 else tag_name.replace("_", " ")
                if tag_name not in undesired_tags:
                    tag_text += ", " + tag_name
            elif i >= len(general_tags) and p >= args.character_threshold:
                tag_name = character_tags[i - len(general_tags)]
                tag_name = tag_name if not args.remove_underscore or len(tag_name) <= 3 else tag_name.replace("_", " ")
                if tag_name not in undesired_tags:
                    tag_text += ", " + tag_name
        tag_text = tag_text[2:] if len(tag_text) > 0 else ''
        captions.append(tag_text)
    return captions

def wd_pass(image_paths, model, args):
    # Preprocess the image
    captions = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = preprocess_image(image)
        captions.append(run_batch([image], model, args))
    return captions

def main():
    args = parse_args()

    # check for the model
    if not os.path.exists(args.model_dir) or args.force_download:
        print(f"downloading wd14 tagger model from hf_hub. id: {args.repo_id}")
        for file in FILES:
            hf_hub_download(args.repo_id, file, cache_dir=args.model_dir, force_download=True)
            
        for file in SUB_DIR_FILES:
            hf_hub_download(
                args.repo_id,
                file,
                subfolder=SUB_DIR,
                cache_dir=os.path.join(args.model_dir, SUB_DIR),
                force_download=True,
                )

    cfg = Config(args)
    model_config = cfg.model_cfg

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)

    model = model.to(torch.device('cuda'))

    vis_processor_cfg = cfg.datasets_cfg.cc_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor)

    chat_state = deepcopy(CONV_VISION)
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
        start_time = time.time()
        caption = describe_image(image_path, chat, chat_state, img_list, num_beams, temperature)

        with open("mycaptions/{}_caption.txt".format(os.path.splitext(os.path.basename(image_path))[0]), "w") as f:
            f.write(caption)
        
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Caption for {os.path.basename(image_path)} saved in 'mycaptions' folder")
        print(f"Time taken to process caption for {os.path.basename(image_path)} is: {time_taken:.2f} s")

    del model  # Unload pytorch model from memory
    torch.cuda.empty_cache()

    # Load Keras model
    keras.backend.clear_session()
    model = load_model(args.model_dir)

    wd_captions = wd_pass(image_paths, model, args)

    for image_path, wd_caption in zip(image_paths, wd_captions):
        wd_caption = wd_caption[0]
        with open("mycaptions/{}_caption.txt".format(os.path.splitext(os.path.basename(image_path))[0]), "a") as f:
            f.write(str(wd_caption))

    del model  # Unload keras model from memory
    keras.backend.clear_session()

if __name__ == '__main__':
    main()