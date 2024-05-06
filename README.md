## Getting Started (LINUX)

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
   wget https://huggingface.co/ckpt/minigpt4/resolve/main/minigpt4.pth -O ./checkpoint.pth
   wget https://huggingface.co/ckpt/minigpt4/resolve/main/blip2_pretrained_flant5xxl.pth -O ./blip2_pretrained_flant5xxl.pth
   ```

   For 7b, then just use this:
   ```
   wget https://huggingface.co/ckpt/minigpt4-7B/resolve/main/prerained_minigpt4_7b.pth -O ./checkpoint.pth
   wget https://huggingface.co/ckpt/minigpt4/resolve/main/blip2_pretrained_flant5xxl.pth -O ./blip2_pretrained_flant5xxl.pth
   ```

To get this right you'd need to replace ./minigpt4/checkpoint.pth with directory your minigpt4 directory + checkpoint.pth, for example. 

3. Install the required packages:

   ```
   pip install cmake
   pip install lit
   pip install -q salesforce-lavis
   pip install -q bitsandbytes
   pip install -q accelerate
   pip install -q git+https://github.com/huggingface/transformers.git -U
   ```

5. Now, you can run the script:

   ```
   python app.py --image-folder path_to_image_folder --beam-search-numbers value
   ```
   
   If you want to test llama 7b then use this:
   
   ```
   python app.py --image-folder path_to_image_folder --beam-search-numbers 2 --model llama7b
   ```
   
In your repository directory you can make two folders namely
```
images  
mycaptions
```

in this case your path_to_image_folder = images
## Features
1. Shows timestamp to process each caption
2. Use --save-in-imgfolder to save captions in your images folder instead.
3. One click setup (setup.bat) for windows.

## To-Do List

- [x] ~~Make it work on Windows~~
- [ ] Implement for MiniGPT-4 7B
- [ ] Include inputs from Segment Anything
- [ ] DOCKER SUPPORT COMING TO YAYYYY


## Acknowledgment

A huge thank you to [Camenduru](https://github.com/camenduru) for developing the awesome MiniGPT-4 Colab, which has served as the foundation for most of this work. Huge thanks to [rafraf](https://www.instagram.com/rafstahelin/) for making the features what they are. This project is primarily aimed at helping people train Stable Diffusion models to mass caption their images.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Check out https://github.com/gessyoo/minigpt4-batch-tweaked fork with implemented changes which removes trivial words like - "The image shows" and "The image is," etc. and the _caption extension from the text captions.
