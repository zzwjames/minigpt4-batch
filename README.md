## Getting Started (LINUX)

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

3. Install the required packages:

   ```
   pip install cmake
   pip install lit
   pip install -q salesforce-lavis
   pip install -q bitsandbytes
   pip install -q accelerate
   pip install -q git+https://github.com/huggingface/transformers.git -U
   ```

5. Run the script:

   ```
   export CUDA_VISIBLE_DEVICES=0
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


