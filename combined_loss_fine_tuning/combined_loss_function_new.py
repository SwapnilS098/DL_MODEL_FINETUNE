import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from compressai.losses import RateDistortionLoss

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
        
        # Load VGG16 and set up layers for multi-scale feature extraction
        vgg = vgg16(pretrained=True).features
        self.feature_layers = nn.ModuleList([nn.Sequential(*list(vgg[:4])),    # First conv block
                                             nn.Sequential(*list(vgg[:9])),    # Second conv block
                                             nn.Sequential(*list(vgg[:16]))])  # Third conv block
        
        # Freeze VGG layers and move to GPU if available
        for layer in self.feature_layers:
            layer.eval()
            if torch.cuda.is_available():
                layer.cuda()

    def compute_cosine_similarity_loss(self, features1, features2):
        """Compute cosine similarity loss between feature maps from target and output."""
        cosine_loss = 0
        for f1, f2 in zip(features1, features2):
            f1_flat = f1.view(f1.size(0), -1)  # Flatten features to [batch, -1]
            f2_flat = f2.view(f2.size(0), -1)
            cosine_sim = F.cosine_similarity(f1_flat, f2_flat, dim=1)
            cosine_loss += (1 - cosine_sim).mean()  # We want to minimize 1 - cosine similarity
        return cosine_loss

    def forward(self, output, target):
        # Ensure input tensors are on the same device as the feature layers
        device = next(self.feature_layers[0].parameters()).device
        target, output_img = target.to(device), output["x_hat"].to(device)

        # Extract multi-scale features for both target and output
        with torch.no_grad():  # No need to compute gradients for feature extraction
            target_features = [layer(target) for layer in self.feature_layers]
            output_features = [layer(output_img) for layer in self.feature_layers]

        # Compute cosine similarity loss for feature preservation
        feature_loss = self.compute_cosine_similarity_loss(target_features, output_features)

        # Calculate the original rate-distortion loss
        rd_loss = self.rate_distortion_loss(output, target)

        # Total loss is the sum of rate-distortion loss and feature preservation loss
        total_loss = rd_loss["loss"] + self.lambda_feature * feature_loss

        # Return the total loss and include components for monitoring
        rd_loss["feature_loss"] = feature_loss.item()
        rd_loss["loss"] = total_loss
        return rd_loss
