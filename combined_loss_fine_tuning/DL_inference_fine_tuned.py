import torch
import torchvision.utils as vutils
from compressai.zoo import models
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

device="cuda" if torch.cuda.is_available() else "cpu"
print("Device is:",device)


def load_model(model_path,device="cpu"):
    """loads the saved model from the path"""

    model_name="bmshj2018-factorized"
    quality=4
    metric="ms-ssim"
    model=models[model_name](quality=quality,metric=metric)
    
    checkpoint=torch.load(model_path,map_location=device)
    print(checkpoint.keys())

    model.load_state_dict(checkpoint["state_dict"])
    
    #model=checkpoint["model"]
    model.eval()
    print("MODEL IS:", model)
    print("model is loaded")
    return model



def inference(model,image_path):
    """
    Runs inference on the input data using the loaded model"""

    #load the image from disc 
    image=Image.open(image_path)
    img_org=image #keep the image save for further use
    #resize and normalize the image
    #using the original image size
    #print("Image size is:",image.size)
    #convert to pytorch tensor and add the batch dimension
    image=transforms.ToTensor()(image).unsqueeze(0).to(device)
    print("Image shape is:",image.shape)
    
    #run inference on the image
    model=model.to(device)  #move the model to the device
    start=time.time()
    with torch.no_grad():
        out_net=model.forward(image)
    out_net['x_hat'].clamp_(0,1)
    end=time.time()
    print("Compression time:",round(end-start,2))   
    
    #Convert this output to the PIL image and then display the image
    output=out_net['x_hat'].squeeze(0).cpu().detach().numpy() #the output is in the range of 0 to 1 and 
    #directly convertig it to uint8 will give the black screen
    output=output*255 # converting it to the range of 0 to 255
    
    print("output type is:",type(output),"shape is:",output.shape)
    output_image=Image.fromarray(output.transpose(1,2,0).astype(np.uint8))
    #output_image.show()
    
    
    #compare the compression quality
    #convert the original image to numpy array
    img_org=np.array(img_org)
    print("img_org shape is:",img_org.shape)
    img_recon=output
    print("initially the recon shape is:",img_recon.shape)
    img_recon=output.transpose(1,2,0)
    print("img_recon shape is:",img_recon.shape)
    print("FINE TUNED MODEL QUALITY")
    ssim_,psnr_=compare_quality(img_org,img_recon)
    #return the output image
    return ssim_,psnr_
    
def org_model_inference(model,image_path):
    """ this is the function for drawing inderence from the 
    original pretrained model
    """
    #load the image from disc 
    image=Image.open(image_path)
    img_org=image #keep the image save for further use
    #resize and normalize the image
    #using the original image size
    #print("Image size is:",image.size)
    #convert to pytorch tensor and add the batch dimension
    image=transforms.ToTensor()(image).unsqueeze(0).to(device)
    print("Image shape is:",image.shape)
    
    
    #run inference on the image
    model_name="bmshj2018-factorized"
    quality=4
    metric="ms-ssim"
    model=models[model_name](quality=quality,metric=metric,pretrained=True).eval().to(device)
    
    model=model.to(device)  #move the model to the device
    start=time.time()
    with torch.no_grad():
        out_net=model.forward(image)
    out_net['x_hat'].clamp_(0,1)
    end=time.time()
    print("Compression time:",round(end-start,2))
    
    #Convert this output to the PIL image and then display the image
    output=out_net['x_hat'].squeeze(0).cpu().detach().numpy() #the output is in the range of 0 to 1 and 
    #directly convertig it to uint8 will give the black screen
    output=output*255 # converting it to the range of 0 to 255
    
    print("output type is:",type(output),"shape is:",output.shape)
    output_image=Image.fromarray(output.transpose(1,2,0).astype(np.uint8))
    #output_image.show()
    
    
    #compare the compression quality
    #convert the original image to numpy array
    img_org=np.array(img_org)
    print("img_org shape is:",img_org.shape)
    img_recon=output
    print("initially the recon shape is:",img_recon.shape)
    img_recon=output.transpose(1,2,0)
    print("img_recon shape is:",img_recon.shape)
    print("ORIGINAL MODEL QUALITY")
    ssim_,psnr_=compare_quality(img_org,img_recon)
    return ssim_,psnr_
    
    
def compare_quality(img_org,img_comp):
    """
    assumes both the images are the same resolution and 
    the images are in RGB format

    Args:
        img_org (numpy array): PIL image converted to numpy array
        img_comp (numpy array): PIL image converted to numpy array
    """
    #convert the images to the float
    #print("img_org:",img_org)
    #print("img_comp:",img_comp)
    #convert the images to the uint8
    img_org=img_org.astype(np.uint8)
    img_comp=img_comp.astype(np.uint8)
    ssim_=ssim(img_org,img_comp,window_size=3,channel_axis=2)
    psnr_=psnr(img_org,img_comp)
    print("SSIM is:",ssim_)
    print("PSNR is:",psnr_)
    return ssim_,psnr_

def quality_compare_on_dataset(model,dataset_path):
    """
    Here a small 5 images dataset is used for the comparision of the quality
    """
    files=os.listdir(dataset_path) 
    images=[]
    for file in files:
        if file.lower().endswith((".png",".jpg",".jpeg",".bmp",".tiff")):
            images.append(file)
    print("Images are:",len(images))
    #Out of these images select only 5
    if len(images)>=5:
        images=images[:5]
        
    print("Images are:",images)
    
    
    #run the inference on the images
    SSIM_org=[]
    PSNR_org=[]
    
    SSIM_fine=[]
    PSNR_fine=[]
    
    for image in images:
        image_path=os.path.join(dataset_path,image)
        
        ssim_,psnr_=org_model_inference(model,image_path)
        SSIM_org.append(ssim_)
        PSNR_org.append(psnr_)
        
        ssim_fine,psnr_fine=inference(model,image_path)
        SSIM_fine.append(ssim_fine)
        PSNR_fine.append(psnr_fine)
        
    print()
    print("SSIM_org:",np.array(SSIM_org).mean())
    print("PSNR_org:",np.array(PSNR_org).mean())
    print()
    print("SSIM_fine:",np.array(SSIM_fine).mean())
    print("PSNR_fine:",np.array(PSNR_fine).mean())
    print()
        

if __name__=="__main__":
    model_path=r"/app/DL_MODEL_FINETUNE/checkpoint_best_loss.pth.tar"
    image_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\DL_MODELFINETUNE\image.png"
    dataset_path=r"/app/datasets/data_50"
    model=load_model(model_path)
    #inference(model,image_path)
    #org_model_inference(model,image_path)
    quality_compare_on_dataset(model,dataset_path)

