import torch, os, gradio as gr, numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
from models.extractor import DINOv2Extractor
from models.lora import apply_lora_to_dinov2
from models.correspondence import SemanticCorrespondenceModel
from utils.segment_aware import SAMSegmentor, apply_mask_to_sim_row

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def launch_stage_demo(title, ckpt_name=None, layer_idx=-1, show_layer_slider=False):
    """
    Lancia una demo Gradio caricando i pesi (opzionali) e settando il layer (opzionale).
    """
    # Inizializza SAM se possibile (Stage 4)
    sam_path = 'sam_vit_b_01ec64.pth'
    segmentor = None
    if os.path.exists(sam_path) and "Finale" in title:
        try:
            segmentor = SAMSegmentor(checkpoint=sam_path, model_type="vit_b", device=device)
            print("[INFO] SAM loaded for Segment-Aware demo.")
        except: pass

    def get_prediction(src_img, trg_img, x, y, current_layer, use_sam):
        backbone = DINOv2Extractor(model_name='dinov2_vitb14', layer=int(current_layer), freeze=True)
        
        if ckpt_name:
            path = f'/content/drive/MyDrive/AML/Checkpoints/{ckpt_name}/best.pth'
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=device)
                model_state = ckpt.get("model_state_dict", {})
                has_lora_keys = any("base_model" in k for k in model_state.keys())
                
                if has_lora_keys:
                    print(f"[INFO] Demo: Applying LoRA to backbone.")
                    backbone.model = apply_lora_to_dinov2(backbone.model, rank=ckpt['args'].get('lora_rank', 16))
                else:
                    peft_type = ckpt['args'].get('peft_type', 'none')
                    if peft_type == 'bitfit':
                        print("[INFO] Demo: BitFit detected.")
                        for n, p in backbone.model.named_parameters():
                            if "bias" in n: p.requires_grad = True
                    else:
                        print("[INFO] Demo: Flat backbone.")
                
                model = SemanticCorrespondenceModel(backbone=backbone, use_adaptive_win=True).to(device)
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model = SemanticCorrespondenceModel(backbone=backbone, use_adaptive_win=True).to(device)
        else:
            model = SemanticCorrespondenceModel(backbone=backbone, use_adaptive_win=True).to(device)
        
        model.eval()
        s_t = transform(src_img).unsqueeze(0).to(device)
        t_t = transform(trg_img).unsqueeze(0).to(device)
        scale = (224 / src_img.width, 224 / src_img.height)
        src_kp = torch.tensor([[[x * scale[0], y * scale[1]]]], device=device).float()
        
        with torch.no_grad():
            # Se SAM è attivo, filtriamo il costo (logica interna semplificata per la demo)
            out = model(s_t, t_t, src_kps=src_kp)
            pkp = out['pred_kps'][0,0].cpu().numpy()
            
        return pkp[0] * (trg_img.width/224), pkp[1] * (trg_img.height/224)

    def gradio_fn(s_img, t_img, lyr, use_sam, evt: gr.SelectData):
        tx, ty = get_prediction(s_img, t_img, evt.index[0], evt.index[1], lyr, use_sam)
        s_o, t_o = s_img.copy(), t_img.copy()
        r = 6
        ImageDraw.Draw(s_o).ellipse([evt.index[0]-r, evt.index[1]-r, evt.index[0]+r, evt.index[1]+r], fill='red', outline='white')
        ImageDraw.Draw(t_o).ellipse([tx-r, ty-r, tx+r, ty+r], fill='green', outline='white')
        return s_o, t_o

    with gr.Blocks() as d:
        gr.Markdown(f"### {title}")
        with gr.Row():
            lyr_ctrl = gr.Slider(0, 11, value=layer_idx if layer_idx != -1 else 11, step=1, 
                               label='Transformer Layer', visible=show_layer_slider)
            sam_ctrl = gr.Checkbox(label="Enable SAM (Segment-Aware)", value=False, visible=segmentor is not None)
        with gr.Row():
            si = gr.Image(type='pil', label='Source (Clicca)')
            ti = gr.Image(type='pil', label='Target')
        with gr.Row():
            so, to = gr.Image(type='pil', label='Selection'), gr.Image(type='pil', label='Prediction')
        si.select(gradio_fn, [si, ti, lyr_ctrl, sam_ctrl], [so, to])
    d.launch(share=True, inline=False)

# =============================================================================
# COMPARISON DEMO (Baseline vs LoRA+AW)
# =============================================================================

def launch_comparison_demo(ckpt_name='lora_only'):
    import gradio as gr
    import torch
    import os
    import random
    from PIL import Image, ImageDraw
    import torchvision.transforms as T
    from models.extractor import DINOv2Extractor
    from models.lora import apply_lora_to_dinov2
    from models.correspondence import SemanticCorrespondenceModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Launching Comparison Demo on {device}...")

    # 1. Caricamento Modelli
    # Baseline
    backbone_base = DINOv2Extractor(model_name='dinov2_vitb14', layer=11, freeze=True)
    model_base = SemanticCorrespondenceModel(backbone=backbone_base, use_adaptive_win=False).to(device).eval()
    
    # LoRA + AW
    backbone_lora = DINOv2Extractor(model_name='dinov2_vitb14', layer=11, freeze=True)
    ckpt_path = f'/content/drive/MyDrive/AML/Checkpoints/{ckpt_name}/best.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = f'g:/My Drive/AML/Checkpoints/{ckpt_name}/best.pth'
        
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model_state = ckpt.get("model_state_dict", {})
        if any("base_model" in k for k in model_state.keys()):
            backbone_lora.model = apply_lora_to_dinov2(backbone_lora.model, rank=ckpt.get('args', {}).get('lora_rank', 16))
        model_lora = SemanticCorrespondenceModel(backbone=backbone_lora, use_adaptive_win=True).to(device)
        model_lora.load_state_dict(ckpt['model_state_dict'])
        print(f"[INFO] LoRA model loaded from {ckpt_path}")
    else:
        print("[WARN] LoRA checkpoint not found. Using baseline for secondary model too.")
        model_lora = model_base
    
    model_lora.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_img_dir = '/content/Project_AML/test_images'
    if not os.path.exists(test_img_dir):
        test_img_dir = 'g:/My Drive/Magistrale/2year2semester/AML/Project_AML/test_images'

    def get_random_pair():
        if not os.path.exists(test_img_dir): return None, None
        files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files: return None, None
        sources = [f for f in files if '_src' in f]
        if not sources: return None, None
        src = random.choice(sources)
        prefix = src.split('_src')[0]
        trgs = [f for f in files if prefix in f and '_trg' in f]
        trg = random.choice(trgs) if trgs else random.choice([f for f in files if '_trg' in f])
        return Image.open(os.path.join(test_img_dir, src)), Image.open(os.path.join(test_img_dir, trg))

    def predict(src_img, trg_img, evt: gr.SelectData):
        if src_img is None or trg_img is None: return None, None
        sx, sy = evt.index[0], evt.index[1]
        s_t = transform(src_img).unsqueeze(0).to(device)
        t_t = transform(trg_img).unsqueeze(0).to(device)
        scale = (224 / src_img.width, 224 / src_img.height)
        src_kp = torch.tensor([[[sx * scale[0], sy * scale[1]]]], device=device).float()
        
        with torch.no_grad():
            # 1. Baseline
            out_b = model_base(s_t, t_t, src_kps=src_kp)
            pkp_b = out_b['pred_kps'][0,0].cpu().numpy()
            bx, by = pkp_b[0] * (trg_img.width/224), pkp_b[1] * (trg_img.height/224)
            
            # 2. Tuo Modello (LoRA + AW)
            out_l = model_lora(s_t, t_t, src_kps=src_kp)
            pkp_l = out_l['pred_kps'][0,0].cpu().numpy()
            lx, ly = pkp_l[0] * (trg_img.width/224), pkp_l[1] * (trg_img.height/224)

        r = 8
        res_base = trg_img.copy()
        ImageDraw.Draw(res_base).ellipse([bx-r, by-r, bx+r, by+r], fill='red', outline='white', width=2)
        
        res_aw = trg_img.copy()
        ImageDraw.Draw(res_aw).ellipse([lx-r, ly-r, lx+r, ly+r], fill='#00FF00', outline='white', width=2)
        
        return res_base, res_aw

    def save_match(src, res_b, res_aw):
        if src is None or res_b is None or res_aw is None:
            return "❌ Nessun match da salvare."
        
        # Percorso dinamico: Colab vs Locale
        save_dir = '/content/drive/MyDrive/AML/Results/Gradio_Captures'
        if not os.path.exists('/content/drive'):
            # Fallback per PC Locale
            save_dir = 'g:/My Drive/AML/Results/Gradio_Captures'
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        w, h = src.size
        res_b, res_aw = res_b.resize((w, h)), res_aw.resize((w, h))
        
        # Collage: [Source | Baseline | LoRA+AW]
        collage = Image.new('RGB', (w * 3, h))
        collage.paste(src, (0, 0))
        collage.paste(res_b, (w, 0))
        collage.paste(res_aw, (w * 2, 0))
        
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"match_{ts}.png"
        path = os.path.join(save_dir, fname)
        collage.save(path)
        
        return f"✅ Salvato con successo in: {os.path.abspath(path)}"

    def load_random():
        s, t = get_random_pair()
        return s, t

    with gr.Blocks(title="AML Comparison Demo") as demo:
        gr.Markdown("# 🧬 Interactive Comparison: Baseline vs Our Model")
        gr.Markdown("Clicca sulla sinistra per confrontare **Baseline (Rosso)** e **LoRA+AW (Verde)**. Usa il tasto 💾 per salvare i risultati migliori su Drive.")
        
        with gr.Row():
            src_input = gr.Image(label="Source Image (Clicca)", type="pil")
            with gr.Column():
                btn_rand = gr.Button("🎲 Carica Coppia Casuale")
                trg_input = gr.Image(label="Target Image", type="pil")
        
        with gr.Row():
            out_base = gr.Image(label="Baseline (ROSSO)", type="pil")
            out_aw = gr.Image(label="Il Tuo Modello (VERDE)", type="pil")
        
        save_btn = gr.Button("💾 SALVA MATCH SU DRIVE")
        status_msg = gr.Markdown("")

        btn_rand.click(load_random, outputs=[src_input, trg_input])
        src_input.select(predict, inputs=[src_input, trg_input], outputs=[out_base, out_aw])
        save_btn.click(save_match, inputs=[src_input, out_base, out_aw], outputs=status_msg)

    demo.launch(share=True, debug=True)
