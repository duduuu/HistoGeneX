import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import timm

class BaseModel(nn.Module):
    def __init__(self, model_name, out_features=3467):
        super(BaseModel, self).__init__()
        
        self.model_name = model_name
        if model_name == "swinv2_s":
            self.model = models.swin_v2_s(pretrained=True)
            self.model.head = nn.Identity()
            in_features = 768
        elif model_name == "swinv2_t":
            self.model = models.swin_v2_t(pretrained=True)
            self.model.head = nn.Identity()
            in_features = 768
        elif model_name == 'convnext_s':
            self.model = models.convnext_small(pretrained=True)
            self.model.classifier[2] = nn.Identity()
            in_features = 768
        elif model_name == 'vit_b_16':
            self.model = models.vit_b_16(pretrained=True)
            self.model.heads = nn.Identity()
            in_features = 768
        elif model_name == 'eva02_s':
            self.model = timm.create_model("eva02_small_patch14_224.mim_in22k", pretrained=True)
            self.model.head = nn.Identity()
            in_features = 384
        elif model_name == 'deit3_s':
            self.model = timm.create_model("deit3_small_patch16_224.fb_in1k", pretrained=True)
            self.model.head = nn.Identity()
            in_features = 384
        elif model_name == 'edgenext_b':
            self.model = timm.create_model("deit3_small_patch16_224.fb_in1k", pretrained=True)
            self.model.head = nn.Identity()
            in_features = 384
        else:
            raise ValueError(f"Unknown model name: {model_name}.")
                        
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU()
        )
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.linear = nn.Linear(384, out_features)
        
    def forward(self, x):
        x = self.model(x)
        #if self.model_name == 'eva02_s':
        #    x = self.feature_extractor(x)
        x = torch.mean(torch.stack([dropout(x) for dropout in self.dropouts]), dim=0)
        x = self.linear(x)
        
        return x
    
class BleepModel(nn.Module):
    def __init__(self, model_name, out_features=3467):
        super().__init__()
        
        if model_name == "swinv2_s":
            self.image_encoder = models.swin_v2_s(pretrained=True)
            self.image_encoder.head = nn.Identity()
            in_features = 768
        else:
            raise ValueError(f"Unknown model name: {model_name}.")
            
        self.image_projection = ProjectionHead(embedding_dim=in_features)
        self.spot_projection = ProjectionHead(embedding_dim=out_features)
        
    def forward(self, imgs, labels):
        # Getting Image and spot Features
        image_features = self.image_encoder(imgs)
        spot_features = labels
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = spot_embeddings @ image_embeddings.T
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax((images_similarity + spots_similarity) / 2, dim=-1)   
        spots_loss = cross_entropy(logits, targets)
        images_loss = cross_entropy(logits.T, targets.T)
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=256, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
def cross_entropy(preds, targets):
    log_softmax = nn.LogSoftmax(dim = -1)
    loss = (-targets * log_softmax(preds)).sum(1)
    
    return loss