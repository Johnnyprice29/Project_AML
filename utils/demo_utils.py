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
    ckpt_path = f'/content/drive/MyDrive/AML/Checkpoints/{ckpt_name}/{ckpt_name}_best.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = f'g:/My Drive/AML/Checkpoints/{ckpt_name}/{ckpt_name}_best.pth'
        
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

    # Uniformiamo il trasform per entrambi i modelli (evita offset di pixel)
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def get_random_spair_pair():
        test_img_dir = './data/SPair-71k/JPEGImages'
        if not os.path.exists(test_img_dir):
            print(f"[DEBUG] SPair JPEGImages not found at {test_img_dir}")
            return None, None
        import glob
        files = []
        for ext in ('*.jpg', '*.png'):
            files.extend(glob.glob(os.path.join(test_img_dir, '**', ext), recursive=True))
        if not files:
            print("[DEBUG] No SPair images found.")
            return None, None
        src_path = random.choice(files)
        cat = os.path.basename(os.path.dirname(src_path))
        trgs = [f for f in files if os.path.basename(os.path.dirname(f)) == cat and f != src_path]
        trg_path = random.choice(trgs) if trgs else src_path
        return Image.open(src_path).convert("RGB"), Image.open(trg_path).convert("RGB")

    def get_random_pfpascal_pair():
        # Try multiple possible roots
        candidates = [
            './data/PF-Pascal',
            '../data/PF-Pascal',
        ]
        # Check for nested subfolder (e.g. PF-Pascal/PF-PASCAL)
        for c in list(candidates):
            if os.path.exists(c):
                subs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d)) and d != '__MACOSX']
                for s in subs:
                    candidates.insert(0, os.path.join(c, s))
        
        root = None
        for c in candidates:
            anno_test = os.path.join(c, "Annotations")
            if not os.path.exists(anno_test):
                anno_test = os.path.join(c, "annotations")
            if os.path.exists(anno_test):
                root = c
                break
        
        if root is None:
            print(f"[DEBUG] PF-Pascal root not found. Tried: {candidates}")
            return None, None
        
        anno_dir = os.path.join(root, "Annotations")
        if not os.path.exists(anno_dir):
            anno_dir = os.path.join(root, "annotations")
        
        print(f"[DEBUG] PF-Pascal root: {root}")
        print(f"[DEBUG] Annotations dir: {anno_dir}")
            
        import glob
        cats = [d for d in os.listdir(anno_dir) if os.path.isdir(os.path.join(anno_dir, d))]
        if not cats:
            print(f"[DEBUG] No category subdirs found in {anno_dir}")
            return None, None
        
        cat = random.choice(cats)
        annos = glob.glob(os.path.join(anno_dir, cat, "*.mat"))
        if len(annos) < 2:
            print(f"[DEBUG] Category '{cat}' has < 2 annotations")
            return None, None
        
        a1, a2 = random.sample(annos, 2)
        i1_name = os.path.basename(a1).replace(".mat", ".jpg")
        i2_name = os.path.basename(a2).replace(".mat", ".jpg")
        
        # Try flat JPEGImages first, then category subfolder
        jpeg_dir = os.path.join(root, "JPEGImages")
        p1 = os.path.join(jpeg_dir, i1_name)
        if not os.path.exists(p1): p1 = os.path.join(jpeg_dir, cat, i1_name)
        p2 = os.path.join(jpeg_dir, i2_name)
        if not os.path.exists(p2): p2 = os.path.join(jpeg_dir, cat, i2_name)
        
        print(f"[DEBUG] Loading img1: {p1} (exists={os.path.exists(p1)})")
        print(f"[DEBUG] Loading img2: {p2} (exists={os.path.exists(p2)})")
        
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")
        return img1, img2

    def predict(src_img, trg_img, evt: gr.SelectData):
        if src_img is None or trg_img is None: return None, None, (0, 0)
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
        
        return res_base, res_aw, (sx, sy)

    def save_match(src, res_b, res_aw, coords):
        if src is None or res_b is None or res_aw is None:
            return "❌ Nessun match da salvare."
        
        sx, sy = coords
        colab_base = '/content/drive/MyDrive' if os.path.exists('/content/drive/MyDrive') else '/content/drive/My Drive'
        save_dir = os.path.join(colab_base, 'AML/Results/Gradio_Captures') if os.path.exists('/content/drive') else 'g:/My Drive/AML/Results/Gradio_Captures'
        os.makedirs(save_dir, exist_ok=True)
        
        w, h = src.size
        src_with_pt = src.copy()
        r = 8
        ImageDraw.Draw(src_with_pt).ellipse([sx-r, sy-r, sx+r, sy+r], fill='yellow', outline='black', width=2)
        
        res_b, res_aw = res_b.resize((w, h)), res_aw.resize((w, h))
        collage = Image.new('RGB', (w * 3, h))
        collage.paste(src_with_pt, (0, 0))
        collage.paste(res_b, (w, 0))
        collage.paste(res_aw, (w * 2, 0))
        
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"match_{ts}.png"
        path = os.path.join(save_dir, fname)
        collage.save(path)
        return f"✅ Salvato con successo in: {os.path.abspath(path)}"

    def load_random():
        s, t = get_random_spair_pair()
        return s, t

    def load_pascal():
        s, t = get_random_pfpascal_pair()
        return s, t

    with gr.Blocks(title="AML Comparison Demo") as demo:
        last_coords = gr.State(value=(0, 0))
        gr.Markdown("# 🧬 Interactive Comparison: Baseline vs Our Model")
        gr.Markdown("Clicca sulla sinistra per confrontare **Baseline (Rosso)** e **LoRA+AW (Verde)**. Usa il tasto 💾 per salvare i risultati migliori su Drive.")
        
        with gr.Row():
            src_input = gr.Image(label="Source Image (Clicca)", type="pil")
            with gr.Column():
                with gr.Row():
                    btn_rand = gr.Button("🎲 Carica da SPair-71k")
                    btn_pascal = gr.Button("🖼️ Carica da PF-Pascal")
                trg_input = gr.Image(label="Target Image", type="pil")
        
        with gr.Row():
            out_base = gr.Image(label="Baseline (ROSSO)", type="pil")
            out_aw = gr.Image(label="Il Tuo Modello (VERDE)", type="pil")
        
        save_btn = gr.Button("💾 SALVA MATCH SU DRIVE")
        status_msg = gr.Markdown("")

        btn_rand.click(load_random, outputs=[src_input, trg_input])
        btn_pascal.click(load_pascal, outputs=[src_input, trg_input])
        src_input.select(predict, inputs=[src_input, trg_input], outputs=[out_base, out_aw, last_coords])
        save_btn.click(save_match, inputs=[src_input, out_base, out_aw, last_coords], outputs=status_msg)

    demo.launch(share=True, debug=True)

# =============================================================================
# ROBUSTNESS DEMO (Geometric Transformations)
# =============================================================================

def launch_robustness_demo(ckpt_name='lora_only'):
    import gradio as gr, torch, os, random
    from PIL import Image, ImageDraw
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from models.extractor import DINOv2Extractor
    from models.lora import apply_lora_to_dinov2
    from models.correspondence import SemanticCorrespondenceModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    backbone = DINOv2Extractor(model_name='dinov2_vitb14', layer=11, freeze=True)
    ckpt_path = f'/content/drive/MyDrive/AML/Checkpoints/{ckpt_name}/{ckpt_name}_best.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model_state = ckpt.get("model_state_dict", {})
        if any("base_model" in k for k in model_state.keys()):
            backbone.model = apply_lora_to_dinov2(backbone.model, rank=ckpt.get('args', {}).get('lora_rank', 16))
        model = SemanticCorrespondenceModel(backbone=backbone, use_adaptive_win=True).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[INFO] Robustness Demo: Loaded {ckpt_path}")
    else:
        model = SemanticCorrespondenceModel(backbone=backbone, use_adaptive_win=True).to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def get_random_pair():
        test_img_dir = './data/SPair-71k/JPEGImages'
        if not os.path.exists(test_img_dir): 
            return None, None
            
        import glob
        files = []
        for ext in ('*.jpg', '*.png'):
            files.extend(glob.glob(os.path.join(test_img_dir, '**', ext), recursive=True))
            
        if not files: return None, None
        
        src_path = random.choice(files)
        cat = os.path.basename(os.path.dirname(src_path))
        trgs = [f for f in files if os.path.basename(os.path.dirname(f)) == cat and f != src_path]
        
        trg_path = random.choice(trgs) if trgs else src_path
        return Image.open(src_path), Image.open(trg_path)

    def get_custom_pair():
        colab_base = '/content/drive/MyDrive' if os.path.exists('/content/drive/MyDrive') else '/content/drive/My Drive'
        custom_dir = os.path.join(colab_base, 'AML/CustomImages') if os.path.exists('/content/drive') else 'g:/My Drive/AML/CustomImages'
        
        if not os.path.exists(custom_dir):
            os.makedirs(custom_dir, exist_ok=True)
            return None, None
            
        import glob
        files = []
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            files.extend(glob.glob(os.path.join(custom_dir, '**', ext), recursive=True))
            
        if not files: return None, None
        
        sources = [f for f in files if '_src' in f.lower()]
        if not sources: 
            src_path = random.choice(files)
            trg_path = random.choice([f for f in files if f != src_path]) if len(files) > 1 else src_path
            return Image.open(src_path), Image.open(trg_path)
            
        src_path = random.choice(sources)
        base = os.path.basename(src_path).lower().replace('_src', '')
        trgs = [f for f in files if '_trg' in f.lower() and base in os.path.basename(f).lower()]
        trg_path = random.choice(trgs) if trgs else random.choice([f for f in files if '_trg' in f.lower()])
        return Image.open(src_path), Image.open(trg_path)

    def load_random():
        s, t = get_random_pair()
        return s, t

    def get_random_pfpascal_pair_robust():
        # Try multiple possible roots
        candidates = [
            './data/PF-Pascal',
            '../data/PF-Pascal',
        ]
        for c in list(candidates):
            if os.path.exists(c):
                subs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d)) and d != '__MACOSX']
                for s in subs:
                    candidates.insert(0, os.path.join(c, s))
        
        root = None
        for c in candidates:
            anno_test = os.path.join(c, "Annotations")
            if not os.path.exists(anno_test):
                anno_test = os.path.join(c, "annotations")
            if os.path.exists(anno_test):
                root = c
                break
        
        if root is None:
            print(f"[DEBUG] PF-Pascal root not found. Tried: {candidates}")
            return None, None
        
        anno_dir = os.path.join(root, "Annotations")
        if not os.path.exists(anno_dir):
            anno_dir = os.path.join(root, "annotations")
        
        import glob
        cats = [d for d in os.listdir(anno_dir) if os.path.isdir(os.path.join(anno_dir, d))]
        if not cats: return None, None
        
        cat = random.choice(cats)
        annos = glob.glob(os.path.join(anno_dir, cat, "*.mat"))
        if len(annos) < 2: return None, None
        
        a1, a2 = random.sample(annos, 2)
        i1_name = os.path.basename(a1).replace(".mat", ".jpg")
        i2_name = os.path.basename(a2).replace(".mat", ".jpg")
        
        jpeg_dir = os.path.join(root, "JPEGImages")
        p1 = os.path.join(jpeg_dir, i1_name)
        if not os.path.exists(p1): p1 = os.path.join(jpeg_dir, cat, i1_name)
        p2 = os.path.join(jpeg_dir, i2_name)
        if not os.path.exists(p2): p2 = os.path.join(jpeg_dir, cat, i2_name)
        
        print(f"[DEBUG] Loading Pascal img1: {p1} (exists={os.path.exists(p1)})")
        print(f"[DEBUG] Loading Pascal img2: {p2} (exists={os.path.exists(p2)})")
        
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")
        return img1, img2

    def load_pascal():
        s, t = get_random_pfpascal_pair_robust()
        return s, t

    def predict_robustness(src_img, trg_img, angle, evt: gr.SelectData):
        if src_img is None or trg_img is None: return None, None, (0,0)
        
        # 1. Original target prediction (0 degrees)
        sx, sy = evt.index[0], evt.index[1]
        s_t = transform(src_img).unsqueeze(0).to(device)
        t_orig = transform(trg_img).unsqueeze(0).to(device)
        scale = (224 / src_img.width, 224 / src_img.height)
        src_kp = torch.tensor([[[sx * scale[0], sy * scale[1]]]], device=device).float()
        
        with torch.no_grad():
            out_orig = model(s_t, t_orig, src_kps=src_kp)
            pkp_orig = out_orig['pred_kps'][0,0].cpu().numpy()
            tx_orig, ty_orig = pkp_orig[0] * (trg_img.width/224), pkp_orig[1] * (trg_img.height/224)

        # 2. Rotated target prediction (angle degrees)
        trg_rot_img = TF.rotate(trg_img, angle, interpolation=T.InterpolationMode.BICUBIC)
        t_rot = transform(trg_rot_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out_rot = model(s_t, t_rot, src_kps=src_kp)
            pkp_rot = out_rot['pred_kps'][0,0].cpu().numpy()
            tx_rot, ty_rot = pkp_rot[0] * (trg_rot_img.width/224), pkp_rot[1] * (trg_rot_img.height/224)

        r = 8
        orig_res = trg_img.copy()
        ImageDraw.Draw(orig_res).ellipse([tx_orig-r, ty_orig-r, tx_orig+r, ty_orig+r], fill='green', outline='white', width=2)
        
        rot_res = trg_rot_img.copy()
        ImageDraw.Draw(rot_res).ellipse([tx_rot-r, ty_rot-r, tx_rot+r, ty_rot+r], fill='yellow', outline='black', width=2)
        
        return orig_res, rot_res, (sx, sy)

    def save_robustness_match(src, res_orig, res_rot, coords):
        if src is None or res_orig is None or res_rot is None:
            return "❌ Nessun match da salvare."
        
        sx, sy = coords
        colab_base = '/content/drive/MyDrive' if os.path.exists('/content/drive/MyDrive') else '/content/drive/My Drive'
        save_dir = os.path.join(colab_base, 'AML/Results/Gradio_Captures') if os.path.exists('/content/drive') else 'g:/My Drive/AML/Results/Gradio_Captures'
        os.makedirs(save_dir, exist_ok=True)
        
        w, h = src.size
        src_with_pt = src.copy()
        r = 8
        ImageDraw.Draw(src_with_pt).ellipse([sx-r, sy-r, sx+r, sy+r], fill='yellow', outline='black', width=2)
        
        res_orig = res_orig.resize((w, h))
        res_rot = res_rot.resize((w, h))
        collage = Image.new('RGB', (w * 3, h))
        collage.paste(src_with_pt, (0, 0))
        collage.paste(res_orig, (w, 0))
        collage.paste(res_rot, (w * 2, 0))
        
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"robustness_{ts}.png"
        path = os.path.join(save_dir, fname)
        collage.save(path)
        return f"✅ Salvato con successo in: {os.path.abspath(path)}"

    with gr.Blocks(title="Geometric Robustness Demo") as demo:
        last_coords = gr.State(value=(0, 0))
        gr.Markdown("# 📐 Interazione Robustezza Geometrica")
        gr.Markdown("Questa demo testa la resilienza del modello alle rotazioni libere. Carica una **coppia reale**, ruota il target e osserva se il modello ritrova il punto corretto.")
        
        with gr.Row():
            src_input = gr.Image(label="Source Image (CLICCA QUI o fai Drop)", type="pil", interactive=True)
            with gr.Column():
                with gr.Row():
                    btn_rand = gr.Button("🎲 Carica da SPair")
                    btn_custom = gr.Button("🖼️ Carica da PF-Pascal")
                trg_input = gr.Image(label="Target Reference (Dropea qui)", type="pil", interactive=True)
                angle_slider = gr.Slider(-180, 180, value=0, step=5, label="Rotazione Target (gradi)")
        
        with gr.Row():
            out_orig = gr.Image(label="Target 0° (Previsione VERDE)", type="pil")
            out_rot = gr.Image(label="Target Ruotato (Previsione GIALLA)", type="pil")
            
        save_btn = gr.Button("💾 SALVA MATCH SU DRIVE")
        status_msg = gr.Markdown("")
        
        btn_rand.click(load_random, outputs=[src_input, trg_input])
        btn_custom.click(load_pascal, outputs=[src_input, trg_input])
        src_input.select(predict_robustness, inputs=[src_input, trg_input, angle_slider], outputs=[out_orig, out_rot, last_coords])
        save_btn.click(save_robustness_match, inputs=[src_input, out_orig, out_rot, last_coords], outputs=status_msg)

    demo.launch(share=True, debug=True)
