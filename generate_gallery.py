import torch
import os
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from models.extractor import DINOv2Extractor
from models.lora import apply_lora_to_dinov2
from models.correspondence import SemanticCorrespondenceModel

print("[START] Inizializzazione galleria visuale...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_prediction(model, src_img, trg_img, px, py):
    s_t = transform(src_img).unsqueeze(0).to(device)
    t_t = transform(trg_img).unsqueeze(0).to(device)
    scale = (224 / src_img.width, 224 / src_img.height)
    src_kp = torch.tensor([[[px * scale[0], py * scale[1]]]], device=device).float()
    with torch.no_grad():
        out = model(s_t, t_t, src_kps=src_kp)
        pkp = out['pred_kps'][0,0].cpu().numpy()
    return pkp[0] * (trg_img.width/224), pkp[1] * (trg_img.height/224)

# Configurazione Coppie e Punti (Coordinate normalizzate 0-1)
pairs = [
    {"name": "Gatto Cross-Domain", "src": "cross_cat_sketch_src.png", "trg": "cross_cat_photo_trg.png", "pts": [(0.4, 0.3), (0.5, 0.5)]},
    {"name": "Leone Statua", "src": "cross_lion_statue_src.png", "trg": "cross_lion_real_trg.png", "pts": [(0.5, 0.4), (0.6, 0.6)]},
    {"name": "Orso Peluche", "src": "cross_bear_toy_src.png", "trg": "cross_bear_real_trg.png", "pts": [(0.5, 0.3), (0.4, 0.5)]},
    {"name": "Aereo Prospettiva", "src": "airplane_src.png", "trg": "airplane_trg.png", "pts": [(0.3, 0.5), (0.7, 0.5)]},
    {"name": "Bici Epoca", "src": "bicycle_src.png", "trg": "bicycle_trg.png", "pts": [(0.5, 0.4), (0.3, 0.7)]},
    {"name": "Cane Azione", "src": "dog_src.png", "trg": "dog_trg.png", "pts": [(0.5, 0.4), (0.6, 0.3)]}
]

img_dir = "g:/My Drive/Magistrale/2year2semester/AML/Project_AML/test_images"
out_dir = "g:/My Drive/AML/Results/Visualizations"
ckpt_path = "g:/My Drive/AML/Checkpoints/lora_only/lora_only_best.pth"

# 1. Caricamento Modelli
print(f"[STEP 1/3] Caricamento modelli (Baseline e LoRA)...")
backbone_base = DINOv2Extractor(model_name='dinov2_vitb14', layer=11, freeze=True)
model_base = SemanticCorrespondenceModel(backbone=backbone_base, use_adaptive_win=False).to(device)
model_base.eval()

backbone_lora = DINOv2Extractor(model_name='dinov2_vitb14', layer=11, freeze=True)
backbone_lora.model = apply_lora_to_dinov2(backbone_lora.model, rank=16)
model_lora = SemanticCorrespondenceModel(backbone=backbone_lora, use_adaptive_win=True).to(device)
if os.path.exists(ckpt_path):
    model_lora.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
    print("[INFO] Checkpoint LoRA caricato con successo.")
model_lora.eval()

# 2. Loop sulle coppie
print(f"[STEP 2/3] Generazione immagini qualitative...")
for p in pairs:
    s_p = os.path.join(img_dir, p["src"])
    t_p = os.path.join(img_dir, p["trg"])
    if not os.path.exists(s_p) or not os.path.exists(t_p): 
        print(f"[WARN] Immagini mancanti per {p['name']}, salto.")
        continue
    
    print(f" -> Elaborazione: {p['name']}...")
    src_img = Image.open(s_p).convert('RGB')
    trg_img = Image.open(t_p).convert('RGB')
    
    # Creiamo un collage: [Source | Baseline | LoRA] per ogni punto
    rows = []
    for (nx, ny) in p["pts"]:
        sx, sy = nx * src_img.width, ny * src_img.height
        bx, by = get_prediction(model_base, src_img, trg_img, sx, sy)
        lx, ly = get_prediction(model_lora, src_img, trg_img, sx, sy)
        
        # Disegno i punti
        r = 12
        s_draw = src_img.copy()
        t_base = trg_img.copy()
        t_lora = trg_img.copy()
        
        ImageDraw.Draw(s_draw).ellipse([sx-r, sy-r, sx+r, sy+r], fill='yellow', outline='black', width=3)
        ImageDraw.Draw(t_base).ellipse([bx-r, by-r, bx+r, by+r], fill='red', outline='white', width=3)
        ImageDraw.Draw(t_lora).ellipse([lx-r, ly-r, lx+r, ly+r], fill='green', outline='white', width=3)
        
        # Concateno orizzontalmente
        row = Image.new('RGB', (src_img.width * 3, src_img.height))
        row.paste(s_draw, (0, 0))
        row.paste(t_base, (src_img.width, 0))
        row.paste(t_lora, (src_img.width * 2, 0))
        rows.append(row)

    # Concateno verticalmente le righe dei punti
    final_img = Image.new('RGB', (src_img.width * 3, src_img.height * len(rows)))
    for i, r_img in enumerate(rows):
        final_img.paste(r_img, (0, i * src_img.height))
    
    final_img.save(os.path.join(out_dir, f"comp_{p['name'].replace(' ', '_').lower()}.png"))

print(f"[STEP 3/3] Gallerie salvate in {out_dir}")
print("[FINISH] Script completato!")
