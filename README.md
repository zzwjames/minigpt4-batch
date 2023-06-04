---
title: MiniGPT-4
emoji: 🚀
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 3.27
app_file: app.py
pinned: false
license: other
---

Welcome to the MiniGPT-4 Batch repo! This repository provides an implementation of MiniGPT-4 to mass caption Stable Diffusion images. It utilizes llama weights that are downloaded automatically if not already present. Please note that this implementation currently works only on Linux systems and runs only on high end machines (not the free colab).

## Getting Started

If you're installing MiniGPT-4 Batch for the first time, please follow these steps:

1. Clone the GitHub repository:

   ```git
   git clone https://github.com/pipinstallyp/minigpt4-batch
   ```
Change directory to minigp4-batch

  ```
   cd minigpt4-batch
   ```
2. Download the necessary files:

   ```
   wget https://huggingface.co/ckpt/minigpt4/resolve/main/minigpt4.pth -O ./minigpt4/checkpoint.pth
   wget https://huggingface.co/ckpt/minigpt4/resolve/main/blip2_pretrained_flant5xxl.pth -O ./minigpt4/blip2_pretrained_flant5xxl.pth
   ```

To get this right you'd need to replace ./minigpt4/checkpoint.pth with directory your minigpt4 directory + checkpoint.pth, for example. 

3. Install the required packages:

   ```
   pip install -q salesforce-lavis
   pip install -q bitsandbytes
   pip install -q accelerate
   pip install -q gradio==3.27.0
   pip install -q git+https://github.com/huggingface/transformers.git -U
   ```

4. Change to the project directory:

   ```
   %cd /minigpt4
   ```

5. Now, you can run the script:

   ```
   python app.py --image-folder path_to_image_folder --beam-search-numbers value
   ```
## To-Do List

- [ ] Make it work on Windows
- [ ] Implement for MiniGPT-4 7B
- [ ] Include inputs from Segment Anything

## Acknowledgment

A huge thank you to [Camenduru](https://github.com/camenduru) for developing the awesome MiniGPT-4 Colab, which has served as the foundation for most of this work. This project is primarily aimed at helping people train Stable Diffusion models to mass caption their images.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
