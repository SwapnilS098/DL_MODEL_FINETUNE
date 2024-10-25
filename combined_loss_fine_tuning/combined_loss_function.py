import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

import numpy as np
#import cv2
#from skimage.feature import match_descriptors
from compressai.losses import RateDistortionLoss
import matplotlib.pyplot as plt

class CombinedLoss(nn.Module):
    def __init__(self, lambda_feature=0.01, lambda_rd=0.01, metric="mse"):
        """
        Initialize the combined loss function.
        - lambda_feature: weight for the feature preservation loss
        - lambda_rd: weight for the rate-distortion loss
        """
        super().__init__()
        self.lambda_feature = lambda_feature
        self.rate_distortion_loss = RateDistortionLoss(lmbda=lambda_rd, metric=metric)
        
        # Initialize FAST detector and ORB descriptor
        #self.fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        #self.orb = cv2.ORB_create()
        
        #Other method for detecting the feature points
        # Load pre-trained VGG16 and extract feature layers
        vgg = vgg16(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg)[:16])  # Use layers until the 3rd conv block
        self.feature_extractor.eval()  # Set to evaluation mode
        
        # Move feature extractor to GPU if available
        if torch.cuda.is_available():
            self.feature_extractor.cuda()
            
    def forward(self, output, target):
        # Ensure input tensors are on the same device as the feature extractor
        device = next(self.feature_extractor.parameters()).device
        target = target.to(device)
        output_img = output["x_hat"].to(device)

        # Extract features using VGG for both target and output
        with torch.no_grad():  # No need to compute gradients for feature extraction
            target_features = self.feature_extractor(target)
            output_features = self.feature_extractor(output_img)

        # Compute feature loss (e.g., L2 norm between feature maps)
        feature_loss = F.mse_loss(target_features, output_features)

        # Calculate the original rate-distortion loss
        rd_loss = self.rate_distortion_loss(output, target)

        # Total loss is the sum of rate-distortion loss and feature preservation loss
        total_loss = rd_loss["loss"] + self.lambda_feature * feature_loss

        # Return the total loss and include components for monitoring
        rd_loss["feature_loss"] = feature_loss.item()
        rd_loss["loss"] = total_loss
        return rd_loss

    """ def forward(self, output, target):
        feature_loss = 0  # Initialize feature preservation loss

        for i in range(target.size(0)):  # Loop over batch size
            # Convert the target image to a NumPy array and grayscale it
            target_img = target[i].cpu().permute(1, 2, 0).detach().numpy()  # Convert to [H, W, C] format
            target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY).astype(np.uint8)
            #plt.imshow(target_img)
            #plt.show()

            # Convert the reconstructed image to a NumPy array and grayscale it
            output_img = output["x_hat"][i].cpu().permute(1, 2, 0).detach().numpy()  # Convert to [H, W, C] format
            output_img_gray = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY).astype(np.uint8)
            #plt.imshow(output_img)
            #plt.show()

            # Check if images are empty
            if target_img_gray.size == 0 or output_img_gray.size == 0:
                print(f"Image {i} is empty.")
                continue

            # Detect keypoints using FAST
            keypoints_original = self.fast.detect(target_img, None)
            keypoints_reconstructed = self.fast.detect(output_img, None)

            # Compute ORB descriptors for both images
            keypoints_original, descriptors_original = self.orb.compute(target_img_gray, keypoints_original)
            keypoints_reconstructed, descriptors_reconstructed = self.orb.compute(output_img_gray, keypoints_reconstructed)

            # Check if descriptors are None
            if descriptors_original is None or descriptors_reconstructed is None:
                print(f"No descriptors found for image {i}")
                continue  # Skip this image if no descriptors are found

            # Match FAST descriptors using OpenCV's BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors_original, descriptors_reconstructed)

            # Calculate the feature preservation loss for the current image
            for match in matches:
                original_keypoint = keypoints_original[match.queryIdx].pt  # Get original keypoint location
                reconstructed_keypoint = keypoints_reconstructed[match.trainIdx].pt  # Get reconstructed keypoint location
                feature_loss += np.linalg.norm(np.array(original_keypoint) - np.array(reconstructed_keypoint))

        # Normalize feature loss (avoid division by zero)
        num_matches = len(matches)
        feature_loss /= num_matches if num_matches > 0 else 1

        # Calculate the original rate-distortion loss
        rd_loss = self.rate_distortion_loss(output, target)

        # Total loss is the sum of rate-distortion loss and feature preservation loss
        total_loss = rd_loss["loss"] + self.lambda_feature * feature_loss

        # Return the total loss and include components for monitoring
        rd_loss["feature_loss"] = feature_loss
        rd_loss["loss"] = total_loss
        return rd_loss """

