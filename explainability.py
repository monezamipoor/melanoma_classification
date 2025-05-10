import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as T
from collections import OrderedDict

from pytorch_grad_cam import (
    LayerCAM,
    GradCAMPlusPlus,
    ShapleyCAM,
    KPCA_CAM,
    FinerCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

model_name="efficientnet_b0"
model_path="checkpoints/2025-04-26_19-51-29-NF_E1-default-jc_AUC.pth"
image_path= "/content/Melanoma/test/ISIC_0509538.jpg"

# 1) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Build a single-logit EfficientNet-B0
model = models.model_name(pretrained=False)
in_feats = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_feats, 1)   # binary head

# 3) Load & unwrap checkpoint
raw = torch.load(model_path, map_location="cpu")

# 4) Clean up the loaded state dict so its keys match torchvision’s naming
cleaned_state_dict = OrderedDict()

for original_key, weight_tensor in sd.items():
    # Start with the raw key name
    new_key = original_key

    # If this checkpoint was wrapped under "backbone.", drop that prefix
    if new_key.startswith("backbone."):
        new_key = new_key[len("backbone."):]

    # If it was further wrapped under "model.", drop that too
    if new_key.startswith("model."):
        new_key = new_key[len("model."):]

    # Store the tensor under the cleaned-up key
    cleaned_state_dict[new_key] = weight_tensor


# 5) Load into model
missing, unexpected = model.load_state_dict(new_sd, strict=False)
print("missing keys:", missing)
print("unexpected keys:", unexpected)

model.to(device).eval()

# 6) Preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]),
])

# 7) CAM function with sigmoid + class index 0
def apply_all_cams_with_finer(image_path, model, transform):
    orig = Image.open(image_path).convert('RGB')
    inp  = transform(orig).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp)                 # [1,1]
        probs  = torch.sigmoid(logits)      # [1,1]
        # only one output, so always target index 0
        top_cl = 0

    target_layer = model.features[-1]

    cam_algos = {
        "Grad-CAM++":   GradCAMPlusPlus(model=model, target_layers=[target_layer]),
        "KPCA-CAM":     KPCA_CAM(model=model, target_layers=[target_layer]),
        "ShapleyCAM":   ShapleyCAM(model=model, target_layers=[target_layer]),
        "FinerCAM":     FinerCAM(model=model, target_layers=[target_layer]),
    }

    results = [("Original", orig)]
    rgb_np  = np.array(orig.resize((224,224))).astype(np.float32) / 255.0

    for title, cam_obj in cam_algos.items():
        cam_map = cam_obj(inp, targets=[ClassifierOutputTarget(top_cl)])[0]
        cam_img = show_cam_on_image(rgb_np, cam_map, use_rgb=True)
        results.append((title, Image.fromarray(cam_img)))

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    for ax, (title, img) in zip(axes, results):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 8) Example usage
apply_all_cams_with_finer(image_path, model, transform)
