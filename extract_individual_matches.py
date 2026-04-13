import torch
import os
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
from models.extractor import DINOv2Extractor
from models.lora import apply_lora_to_dinov2
from models.correspondence import SemanticCorrespondenceModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_predictions(model, src_img, trg_img, points):
    s_t = transform(src_img).unsqueeze(0).to(device)
    t_t = transform(trg_img).unsqueeze(0).to(device)
    scale = (224 / src_img.width, 224 / src_img.height)
    src_kps = torch.tensor([[ [p[0]*scale[0], p[1]*scale[1]] for p in points ]], device=device).float()
    with torch.no_grad():
        out = model(s_t, t_t, src_kps=src_kps)
        pred_kps = out['pred_kps'][0].cpu().numpy()
    return [(pkp[0] * (trg_img.width/224), pkp[1] * (trg_img.height/224)) for pkp in pred_kps]

pairs = [
    {"id": "cat", "src": "cross_cat_sketch_src.png", "trg": "cross_cat_photo_trg.png"},
    {"id": "lion", "src": "cross_lion_statue_src.png", "trg": "cross_lion_real_trg.png"},
    {"id": "bear", "src": "cross_bear_toy_src.png", "trg": "cross_bear_real_trg.png"},
    {"id": "airplane", "src": "airplane_src.png", "trg": "airplane_trg.png"},
    {"id": "dog", "src": "dog_src.png", "trg": "dog_trg.png"},
    {"id": "bike", "src": "bicycle_src.png", "trg": "bicycle_trg.png"},
    {"id": "bird", "src": "real_bird_src.jpg", "trg": "real_bird_trg.jpg"},
    {"id": "car", "src": "real_car_src.jpg", "trg": "real_car_trg.jpg"}
]

img_dir = "g:/My Drive/Magistrale/2year2semester/AML/Project_AML/test_images"
out_dir = "g:/My Drive/AML/Results/Curated_Matches"
ckpt_path = "g:/My Drive/AML/Checkpoints/lora_only/lora_only_best.pth"

backbone_base = DINOv2Extractor(model_name='dinov2_vitb14', layer=11, freeze=True)
model_base = SemanticCorrespondenceModel(backbone=backbone_base, use_adaptive_win=False).to(device).eval()
backbone_lora = DINOv2Extractor(model_name='dinov2_vitb14', layer=11, freeze=True)
backbone_lora.model = apply_lora_to_dinov2(backbone_lora.model, rank=16)
model_lora = SemanticCorrespondenceModel(backbone=backbone_lora, use_adaptive_win=True).to(device)
model_lora.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
model_lora.eval()

for p in pairs:
    s_p, t_p = os.path.join(img_dir, p["src"]), os.path.join(img_dir, p["trg"])
    if not os.path.isfile(s_p): continue
    src_img, trg_img = Image.open(s_p).convert('RGB'), Image.open(t_p).convert('RGB')
    
    # 6 punti sparsi per ogni immagine
    pts = []
    for x in np.linspace(0.3, 0.7, 3):
        for y in np.linspace(0.3, 0.7, 2):
            pts.append((x * src_img.width, y * src_img.height))
            
    base_preds = get_predictions(model_base, src_img, trg_img, pts)
    lora_preds = get_predictions(model_lora, src_img, trg_img, pts)
    
    for i, (sx, sy) in enumerate(pts):
        bx, by = base_preds[i]
        lx, ly = lora_preds[i]
        
        r = 10
        s_d, t_b, t_l = src_img.copy(), trg_img.copy(), trg_img.copy()
        ImageDraw.Draw(s_d).ellipse([sx-r, sy-r, sx+r, sy+r], fill='yellow', outline='black', width=3)
        ImageDraw.Draw(t_b).ellipse([bx-r, by-r, bx+r, by+r], fill='red', outline='white', width=3)
        ImageDraw.Draw(t_l).ellipse([lx-r, ly-r, lx+r, ly+r], fill='#00FF00', outline='white', width=3)
        
        # Salvataggio individuale
        indiv = Image.new('RGB', (src_img.width * 3, src_img.height))
        indiv.paste(s_d, (0, 0))
        indiv.paste(t_b, (src_img.width, 0))
        indiv.paste(t_l, (src_img.width * 2, 0))
        indiv.save(os.path.join(out_dir, f"{p['id']}_pt_{i+1}.png"))

print(f"Estrazione completata: {out_dir}")
