import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import numpy
import os
import sys
import numpy
import torch
import rembg
import threading
import urllib.request
from PIL import Image
import streamlit as st
import huggingface_hub
from app import SAMAPI
def segment_img(img: Image):
    output = rembg.remove(img)
    mask = numpy.array(output)[:, :, 3] > 0
    sam_mask = SAMAPI.segment_api(numpy.array(img)[:, :, :3], mask)
    segmented_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    segmented_img.paste(img, mask=Image.fromarray(sam_mask))
    return segmented_img


def segment_6imgs(zero123pp_imgs):
    imgs = [zero123pp_imgs.crop([0, 0, 320, 320]),
            zero123pp_imgs.crop([320, 0, 640, 320]),
            zero123pp_imgs.crop([0, 320, 320, 640]),
            zero123pp_imgs.crop([320, 320, 640, 640]),
            zero123pp_imgs.crop([0, 640, 320, 960]),
            zero123pp_imgs.crop([320, 640, 640, 960])]
    segmented_imgs = []
    import numpy as np
    for i, img in enumerate(imgs):
        output = rembg.remove(img)
        mask = numpy.array(output)[:, :, 3]
        mask = SAMAPI.segment_api(numpy.array(img)[:, :, :3], mask)
        data = numpy.array(img)[:,:,:3]
        data2 = numpy.ones([320,320,4])
        data2[:,:,:3] = data
        for i in np.arange(data2.shape[0]):
                for j in np.arange(data2.shape[1]):
                        if mask[i,j]==1:
                                data2[i,j,3]=255
        segmented_imgs.append(data2)

        #torch.manual_seed(42)
    return segmented_imgs

def process_img(path,destination,pipeline, is_first):
    # Download an example image.
        print('processing:',path)
        #cond_whole = Image.open('output.png')
        cond = Image.open(path)
        # Run the pipeline!
        result = pipeline(cond, num_inference_steps=75,is_first = is_first).images[0]
        # for general real and synthetic images of general objects
        # usually it is enough to have around 28 inference steps
        # for images with delicate details like faces (real or anime)
        # you may need 75-100 steps for the details to construct

        #result.show()
        #result.save("./test_png/zero123pp/output.png")
        result=segment_6imgs(result)
        print('saving:',os.path.join(destination,'0~5.png'),'in',destination)
        for i in numpy.arange(6):
            Image.fromarray(numpy.uint8(result[i])).save(os.path.join(destination,'{}.png'.format(i)))



if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to process")
    # DiffusionPipeline.from_pretrained cannot received relative path for custom pipeline
    parser.add_argument("--pipeline_path", required=True, help="path of pipeline code, in ../guidance/zero123pp")
    args, extras = parser.parse_known_args()


    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1", custom_pipeline=args.pipeline_path,
        torch_dtype=torch.float16
    )

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    pipeline.to('cuda:0')
    
    
    directory = args.path+'/'
    os.makedirs(directory+'ref', exist_ok=True)
    os.system(f"cp -r {directory+'*.png'} {directory+'ref/'}")
    is_first = True
    l=sorted(os.listdir(directory+'ref'))
        

    for file in  sorted(os.listdir(directory+'ref')):
        if  file[-4:-1]=='.pn':
                
            filename =  os.path.splitext(os.path.basename(file))[0]
            destination = os.path.join(directory+'zero123',filename)
            
            os.makedirs(destination, exist_ok=True)
            img_path = os.path.join(directory+'ref',file)
            process_img(img_path,destination,pipeline, is_first)
            is_first = False

    
