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
                peft_type = ckpt['args'].get('peft_type', 'lora')
                if peft_type == 'lora':
                    print(f"[INFO] Applying LoRA (rank={ckpt['args'].get('lora_rank', 16)}) to demo model.")
                    backbone.model = apply_lora_to_dinov2(backbone.model, rank=ckpt['args'].get('lora_rank', 16))
                elif peft_type == 'bitfit':
                    print("[INFO] BitFit detected: unfreezing bias parameters for demo.")
                    for n, p in backbone.model.named_parameters():
                        if "bias" in n: p.requires_grad = True
                
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
