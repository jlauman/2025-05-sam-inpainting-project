from PIL import Image
import requests
from transformers import SamModel, SamProcessor
from diffusers import DiffusionPipeline, AutoPipelineForText2Image, AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

import torch
import numpy as np

import app


model = SamModel.from_pretrained(
    "facebook/sam-vit-large", 
    torch_dtype=torch.float32
)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"{device=}")
model.to(device=device, dtype=torch.float32)

# Load the SamProcessor using the facebook/sam-vit-base checkpoint
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


################################################################################ 


def mask_to_rgb(mask):
    """
    Transforms a binary mask into an RGBA image for visualization
    """
    bg_transparent = np.zeros(mask.shape + (4, ), dtype=np.uint8)
    
    # Color the area we will replace in green
    # (this vector is [Red, Green, Blue, Alpha])
    bg_transparent[mask == 1] = [0, 255, 0, 127]
    
    return bg_transparent


def get_processed_inputs(image, input_points):
    """
    Use the processor to generate the right inputs for SAM.
    Use "image" as your image.
    Use 'input_points' as your input_points,
    and remember to use the option return_tensors='pt'.
    Also, remember to add .to("cuda") at the end.
    """
    inputs = processor(
        images=image, 
        input_points=input_points, 
        return_tensors="pt"
    ).to(device, dtype=torch.float32)

    # Call SAM
    with torch.no_grad():
        outputs = model(**inputs)

    # print(f"{outputs=}")
    
    # Now let's post process the outputs of SAM to obtain the masks
    masks = processor.image_processor.post_process_masks(
       outputs.pred_masks.cpu(), 
       inputs["original_sizes"].cpu(), 
       inputs["reshaped_input_sizes"].cpu()
    )
    
    # Here we select the mask with the highest score
    # as the mask we will use. You can experiment with also
    # other selection criteria, for example the largest mask
    # instead of the most confident mask
    best_mask = masks[0][0][outputs.iou_scores.argmax()] 

    # NOTE: we invert the mask by using the ~ operator because
    # so that the subject pixels will have a value of 0 and the
    # background pixels a value of 1. This will make it more convenient
    # to infill the background
    return ~best_mask.cpu().numpy()


################################################################################ 


pipeline = AutoPipelineForInpainting.from_pretrained(
    # "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,
    variant=None,
).to(device)

pipeline.enable_attention_slicing()


################################################################################ 


def inpaint(raw_image, input_mask, prompt, negative_prompt=None, seed=74294536, cfgs=7):
    
    mask_image = Image.fromarray(input_mask)
    
    rand_gen = torch.manual_seed(seed)
    
    # Use the pipeline we have created in the previous cell.
    # Use "prompt" as prompt,  "negative_prompt" as the negative prompt,
    # raw_image as the image, mask_image as the mask_image,
    # rand_gen as the generator and cfgs as the guidance_scale.
    
    image = pipeline(
        prompt=prompt,
        negative_propt=negative_prompt,
        image=raw_image,
        mask_image=mask_image,
        generator=rand_gen,
        guidance_scale=cfgs,
        device=device,
    ).images[0]
    
    return image


################################################################################ 


if __name__ == "__main__":
    my_app = app.generate_app(get_processed_inputs, inpaint)