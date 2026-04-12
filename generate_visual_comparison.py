import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
from models.extractor import DINOv2Extractor
from models.lora import apply_lora_to_dinov2
from models.correspondence import SemanticCorrespondenceModel
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_kps(model, src_img, trg_img, x, y):
    s_t = transform(src_img).unsqueeze(0).to(device)
    t_t = transform(trg_img).unsqueeze(0).to(device)
    scale = (224 / src_img.width, 224 / src_img.height)
    src_kp = torch.tensor([[[x * scale[0], y * scale[1]]]], device=device).float()
    with torch.no_grad():
        out = model(s_t, t_t, src_kps=src_kp)
        pkp = out['pred_kps'][0,0].cpu().numpy()
    return pkp[0] * (trg_img.width/224), pkp[1] * (trg_img.height/224)

# 1. Carico Immagini (Gatto Schizzo vs Foto)
src_path = "g:/My Drive/Magistrale/2year2semester/AML/Project_AML/test_images/cross_cat_sketch_src.png"
trg_path = "g:/My Drive/Magistrale/2year2semester/AML/Project_AML/test_images/cross_cat_photo_trg.png"
if not os.path.exists(src_path): exit()

src = Image.open(src_path).convert('RGB')
trg = Image.open(trg_path).convert('RGB')

# 2. Modello Baseline
backbone_base = DINOv2Extractor(model_name='dinov2_vitb14', layer=11, freeze=True)
model_base = SemanticCorrespondenceModel(backbone=backbone_base, use_adaptive_win=False).to(device)
model_base.eval()

# 3. Modello LoRA + AW
ckpt_path = "g:/My Drive/AML/Checkpoints/lora_only/lora_only_best.pth"
backbone_lora = DINOv2Extractor(model_name='dinov2_vitb14', layer=11, freeze=True)
ckpt = torch.load(ckpt_path, map_location=device)
backbone_lora.model = apply_lora_to_dinov2(backbone_lora.model, rank=16)
model_lora = SemanticCorrespondenceModel(backbone=backbone_lora, use_adaptive_win=True).to(device)
model_lora.load_state_dict(ckpt['model_state_dict'])
model_lora.eval()

# 4. Predizione su un punto (es: Orecchio del gatto)
sx, sy = 250, 150 # Coordinata indicativa per l'orecchio
bx, by = get_kps(model_base, src, trg, sx, sy)
lx, ly = get_kps(model_lora, src, trg, sx, sy)

# 5. Creazione composizione
final_w = src.width * 2
final_h = src.height
comp = Image.new('RGB', (final_w, final_h * 2))

# Prima riga: Baseline
draw_s = src.copy()
draw_t_b = trg.copy()
r = 10
idx = (sx, sy)
ImageDraw.Draw(draw_s).ellipse([sx-r, sy-r, sx+r, sy+r], fill='red')
ImageDraw.Draw(draw_t_b).ellipse([bx-r, by-r, bx+r, by+r], fill='red')
comp.paste(draw_s, (0, 0))
comp.paste(draw_t_b, (src.width, 0))

# Seconda riga: LoRA + AW
draw_t_l = trg.copy()
ImageDraw.Draw(draw_t_l).ellipse([lx-r, ly-r, lx+r, ly+r], fill='green', outline='white', width=3)
comp.paste(draw_s, (0, final_h))
comp.paste(draw_t_l, (src.width, final_h))

# Save
comp.save('g:/My Drive/AML/Results/visual_comparison.png')
print("Confronto visivo salvato in Results/visual_comparison.png")
