import os
import sys
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap
import torch
from train_asym import TrainerAsym
from utils.get_args import get_args

ACCENT = (42, 106, 243)
BORDER = (0, 0, 0)
BG = (255, 255, 255)
TILE_BG = (246, 246, 246)
GUTTER = 12
WRAP_TOP = 26
WRAP_QUERY = 40
MAX_TOP_LINES = 3
MAX_QUERY_LINES = 5
LINE_SPACING = 4

def _to_pil(img_tensor):
    arr = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

# CLIP normalization constants used in BaseDataset
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def _to_pil_denorm(img_tensor):
    x = img_tensor.detach().cpu().numpy()  # [C,H,W], normalized
    x = (x.transpose(1,2,0) * CLIP_STD + CLIP_MEAN).clip(0.0, 1.0)
    x = (x * 255).astype(np.uint8)
    return Image.fromarray(x)

def _annotate_badge(img, rank_text, sim_text):
    im = img.copy()
    d = ImageDraw.Draw(im)
    try:
        f_small = _get_font(20)
        f_label = _get_font(22)
    except Exception:
        f_small = None
        f_label = None
    bx = 10
    by = 10
    d.ellipse([(bx, by), (bx + 36, by + 36)], fill=ACCENT)
    # rank (white with dark shadow)
    d.text((bx + 11, by + 8), rank_text, fill=BORDER, font=f_small)
    d.text((bx + 10, by + 7), rank_text, fill=BG, font=f_small)
    # sim (white with dark shadow)
    sx, sy = bx + 48, by + 8
    d.text((sx + 1, sy + 1), sim_text, fill=BORDER, font=f_label)
    d.text((sx, sy), sim_text, fill=BG, font=f_label)
    return im

def _pseudocolor(img):
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 3:
        gray = 0.299 * arr[...,0] + 0.587 * arr[...,1] + 0.114 * arr[...,2]
    else:
        gray = arr
    v = np.clip(gray, 0.0, 1.0)
    # approximate jet colormap
    def jet(x):
        r = np.clip(1.5 - np.abs(4*x - 3), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4*x - 2), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4*x - 1), 0.0, 1.0)
        return np.stack([r, g, b], axis=-1)
    rgb = (jet(v) * 255).astype(np.uint8)
    return Image.fromarray(rgb)

def _get_font(size):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

def _wrap_text(txt, width, max_lines):
    lines = textwrap.wrap(txt, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        # add ellipsis to last line
        if lines:
            lines[-1] = lines[-1].rstrip('.') + '…'
    return lines

def _measure_block(draw, font, lines):
    if not lines:
        return 0
    # approximate line height using bbox of sample
    try:
        _, _, _, h = draw.textbbox((0,0), "Ag", font=font)
        line_h = h
    except Exception:
        line_h = font.getsize("Ag")[1]
    total_h = len(lines) * (line_h + LINE_SPACING)
    return total_h

def _first_caption(ds, idx):
    cap = ds.captions[idx]
    txt = ""
    if isinstance(cap, (list, tuple)):
        txt = str(cap[0]) if cap else ""
    elif isinstance(cap, np.ndarray):
        if cap.size > 0:
            try:
                txt = str(cap.flat[0])
            except Exception:
                txt = ""
    else:
        txt = str(cap or "")
    if isinstance(txt, bytes):
        try:
            txt = txt.decode("utf-8", errors="ignore")
        except Exception:
            txt = str(txt)
    return " ".join(txt.split())

def _letterbox(img, size, bg=None):
    if bg is None:
        bg = TILE_BG
    w, h = img.size
    scale = min(size / w, size / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    img_rs = img.resize((nw, nh))
    canvas = Image.new('RGB', (size, size), bg)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas.paste(img_rs, (x, y))
    return canvas

def _make_grid(images, rows, cols, size):
    total_w = cols * size + (cols - 1) * GUTTER
    total_h = rows * size + (rows - 1) * GUTTER
    grid = Image.new('RGB', (total_w, total_h), TILE_BG)
    for i, img in enumerate(images[:rows * cols]):
        img_tile = _letterbox(img, size, bg=(250, 250, 250))
        r = i // cols
        c = i % cols
        x = c * (size + GUTTER)
        y = r * (size + GUTTER)
        grid.paste(img_tile, (x, y))
    return grid

def _compose_panel(query_img, top_images, scores, title, size=240, query_text="", top_texts=None, subtitle=""):
    rows = 1
    cols = len(top_images)
    grid = _make_grid(top_images, rows, cols, size)
    qimg = _letterbox(query_img, size, bg=TILE_BG)
    pad = 60
    # fonts
    title_font = _get_font(30)
    label_font = _get_font(26)
    small_font = _get_font(24)
    # pre-wrap texts and measure
    tmp = Image.new('RGB', (10,10))
    tmp_draw = ImageDraw.Draw(tmp)
    q_lines = _wrap_text(query_text or "", width=WRAP_QUERY, max_lines=MAX_QUERY_LINES)
    q_card_h = _measure_block(tmp_draw, small_font, q_lines) + 24
    top_blocks = []
    max_top_card_h = 0
    if top_texts:
        for tt in top_texts:
            lines = _wrap_text(tt or "", width=WRAP_TOP, max_lines=MAX_TOP_LINES)
            h = _measure_block(tmp_draw, small_font, lines) + 24
            max_top_card_h = max(max_top_card_h, h)
            top_blocks.append((lines, h))
    text_area_h = max(q_card_h, max_top_card_h) + 20
    header_h = 42
    panel_w = size * (cols + 1) + GUTTER * cols + pad * 2
    panel_h = header_h + size + pad * 2 + text_area_h
    panel = Image.new('RGB', (panel_w, panel_h), BG)
    draw = ImageDraw.Draw(panel)
    # header
    draw.text((pad, 12), title, fill=BORDER, font=title_font)
    if subtitle:
        draw.text((pad + 420, 12), subtitle, fill=ACCENT, font=label_font)
    # tiles
    panel.paste(qimg, (pad, header_h))
    panel.paste(grid, (pad + size + GUTTER, header_h))
    draw.rectangle([(pad - 2, header_h - 2), (pad + size + 2, header_h + size + 2)], outline=BORDER, width=2)
    for i, s in enumerate(scores):
        # ranking badge
        bx = pad + size + GUTTER + i * (size + GUTTER) + 10
        by = header_h + 10
        draw.ellipse([(bx, by), (bx + 34, by + 34)], fill=ACCENT)
        draw.text((bx + 8, by + 5), f"{i+1}", fill=BG, font=_get_font(20))
        # similarity under tile
        tx = pad + size + GUTTER + i * (size + GUTTER) + 12
        ty = header_h + size + 8
        draw.text((tx, ty), f"sim={s:.3f}", fill=BORDER, font=label_font)
    if q_lines:
        q_block = "\n".join(q_lines)
        x0 = pad - 4
        y0 = header_h + size + 48
        draw.rectangle([(x0, y0), (x0 + size + 8, y0 + q_card_h)], fill=(245, 245, 245))
        draw.text((x0 + 8, y0 + 8), q_block, fill=BORDER, font=small_font, spacing=LINE_SPACING)
    if top_blocks:
        for i, (lines, h) in enumerate(top_blocks):
            if not lines:
                continue
            tx = pad + size + GUTTER + i * (size + GUTTER) + 6
            ty = header_h + size + 36
            draw.rectangle([(tx - 4, ty - 4), (tx + size - 8, ty + h)], fill=(245, 245, 245))
            draw.text((tx, ty + 8), "\n".join(lines), fill=BORDER, font=small_font, spacing=LINE_SPACING)
    return panel

def _save_t2i(trainer, k_bits, topk, out_dir, size=240, max_queries=6):
    qi, qt = trainer.get_code(trainer.query_loader, trainer.args.query_num)
    ri, rt = trainer.get_code(trainer.retrieval_loader, trainer.args.retrieval_num)
    q = qt[k_bits].to(trainer.device)
    r = ri[k_bits].to(trainer.device)
    sims = torch.matmul(q, r.t()).cpu().numpy()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for qidx in range(min(max_queries, q.shape[0])):
        order = np.argsort(-sims[qidx])[:topk]
        imgs = []
        scores = []
        tops_txt = []
        for ridx in order:
            img_t = trainer.retrieval_loader.dataset._load_image(int(ridx))
            imgs.append(_to_pil(img_t))
            scores.append(float(sims[qidx, ridx]))
            tops_txt.append(_first_caption(trainer.retrieval_loader.dataset, int(ridx)))
        qimg_t = trainer.query_loader.dataset._load_image(int(qidx))
        qimg = _to_pil(qimg_t)
        q_text = _first_caption(trainer.query_loader.dataset, int(qidx))
        subtitle = f"dataset={trainer.args.dataset}  tag={trainer.args.exp_tag}"
        panel = _compose_panel(qimg, imgs, scores, title=f"Text->Image  bits={k_bits}", query_text=q_text, top_texts=tops_txt, subtitle=subtitle)
        panel.save(os.path.join(out_dir, f't2i_bits{k_bits}_q{qidx}.jpg'))

def _save_i2t(trainer, k_bits, topk, out_dir, size=240, max_queries=6):
    qi, qt = trainer.get_code(trainer.query_loader, trainer.args.query_num)
    ri, rt = trainer.get_code(trainer.retrieval_loader, trainer.args.retrieval_num)
    q = qi[k_bits].to(trainer.device)
    r = rt[k_bits].to(trainer.device)
    sims = torch.matmul(q, r.t()).cpu().numpy()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for qidx in range(min(max_queries, q.shape[0])):
        order = np.argsort(-sims[qidx])[:topk]
        texts = []
        scores = []
        for ridx in order:
            txt = _first_caption(trainer.retrieval_loader.dataset, int(ridx))
            texts.append(txt)
            scores.append(float(sims[qidx, ridx]))
        qimg_t = trainer.query_loader.dataset._load_image(int(qidx))
        qimg = _to_pil(qimg_t)
        width = size * (topk + 1) + 200
        height = size + 160
        panel = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        title_font = _get_font(28)
        label_font = _get_font(24)
        small_font = _get_font(22)
        qimg_rs = qimg.resize((size, size))
        panel.paste(qimg_rs, (20, 20))
        draw.rectangle([(18, 18), (20 + size + 2, 20 + size + 2)], outline=(0, 0, 0), width=2)
        q_text = _first_caption(trainer.query_loader.dataset, int(qidx))
        subtitle = f"dataset={trainer.args.dataset}  tag={trainer.args.exp_tag}"
        draw.text((20, 20 + size + 10), f"Image->Text  bits={k_bits}", fill=BORDER, font=title_font)
        draw.text((20 + 340, 20 + size + 10), subtitle, fill=ACCENT, font=label_font)
        q_lines = textwrap.wrap(q_text, width=36)
        draw.text((20, 20 + size + 44), "\n".join(q_lines[:4]), fill=(0, 0, 0), font=small_font)
        x0 = 20 + size + 20
        y0 = 20
        for i in range(topk):
            item = texts[i] if i < len(texts) else ""
            simv = scores[i] if i < len(scores) else 0.0
            draw.ellipse([(x0 - 6, y0 + i * 36 + 2), (x0 + 22, y0 + i * 36 + 30)], fill=ACCENT)
            draw.text((x0, y0 + i * 36 + 4), f"{i+1}", fill=BG, font=_get_font(20))
            draw.text((x0 + 28, y0 + i * 36), f"sim={simv:.3f}  {item[:120]}", fill=BORDER, font=label_font)
        panel.save(os.path.join(out_dir, f'i2t_bits{k_bits}_q{qidx}.jpg'))

def _save_composite(trainer, k_list, out_dir, topk=5, queries=3, size=240):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    panels = []
    for k_bits in k_list:
        # take first N queries for t2i
        qi, qt = trainer.get_code(trainer.query_loader, trainer.args.query_num)
        ri, rt = trainer.get_code(trainer.retrieval_loader, trainer.args.retrieval_num)
        q = qt[k_bits].to(trainer.device)
        r = ri[k_bits].to(trainer.device)
        sims = torch.matmul(q, r.t()).cpu().numpy()
        for qidx in range(min(queries, q.shape[0])):
            order = np.argsort(-sims[qidx])[:topk]
            imgs = []
            scores = []
            tops_txt = []
            for ridx in order:
                img_t = trainer.retrieval_loader.dataset._load_image(int(ridx))
                imgs.append(_to_pil(img_t))
                scores.append(float(sims[qidx, ridx]))
                tops_txt.append(_first_caption(trainer.retrieval_loader.dataset, int(ridx)))
            qimg_t = trainer.query_loader.dataset._load_image(int(qidx))
            qimg = _to_pil(qimg_t)
            q_text = _first_caption(trainer.query_loader.dataset, int(qidx))
            subtitle = f"dataset={trainer.args.dataset}  tag={trainer.args.exp_tag}"
            panel = _compose_panel(qimg, imgs, scores, title=f"Text->Image  bits={k_bits}", query_text=q_text, top_texts=tops_txt, subtitle=subtitle)
            panels.append(panel)
    # stitch panels: columns=len(k_list), rows=queries
    cols = len(k_list)
    rows = queries
    if not panels:
        return
    pw, ph = panels[0].size
    comp = Image.new('RGB', (cols * pw + (cols - 1) * GUTTER, rows * ph + (rows - 1) * GUTTER), BG)
    for idx, p in enumerate(panels[:rows * cols]):
        r = idx // cols
        c = idx % cols
        x = c * (pw + GUTTER)
        y = r * (ph + GUTTER)
        comp.paste(p, (x, y))
    comp.save(os.path.join(out_dir, 't2i_composite.jpg'))

def _select_interesting_indices(ds, max_queries, dataset_name):
    indices = []
    # Keywords for positive findings
    dataset_name = dataset_name.lower()
    if 'xray' in dataset_name or 'mimic' in dataset_name:
        # High priority: explicit diseases
        pos_keywords = ['cardiomegaly', 'opacity', 'edema', 'atelectasis', 'effusion', 'pneumothorax', 'consolidation', 'fracture', 'hernia', 'nodule', 'mass', 'tortuous', 'enlarged', 'calcified']
    elif 'odir' in dataset_name:
        pos_keywords = ['retinopathy', 'glaucoma', 'cataract', 'degeneration', 'myopia', 'hypertension', 'bleeding', 'drusen', 'laser']
    else:
        # For other datasets, just return the first N
        return list(range(min(len(ds), max_queries)))

    # Scan dataset
    candidates = []
    count = 0
    # Scan up to all samples
    for i in range(len(ds)):
        if count >= max_queries:
            break

        cap = _first_caption(ds, i).lower()

        # Check positive keywords
        # We want to prioritize samples that have these keywords.
        # We also want to avoid "no " preceding the keyword if possible, but that's hard with simple matching.
        # However, finding ANY of these words is usually better than "normal".

        has_pos = any(k in cap for k in pos_keywords)

        # Heuristic: if it has positive keywords, we take it.
        # We can also check if it explicitly says "normal" or "no acute" and exclude those if we want strictly abnormal.
        # But let's just prioritize presence of findings.

        if has_pos:
            candidates.append(i)
            count += 1

    # If not enough, fill with others
    if len(candidates) < max_queries:
        for i in range(len(ds)):
            if len(candidates) >= max_queries:
                break
            if i not in candidates:
                candidates.append(i)

    return candidates

def _save_t2i_raw(trainer, k_bits, topk, out_dir, max_queries=6):
    qi, qt = trainer.get_code(trainer.query_loader, trainer.args.query_num)
    ri, rt = trainer.get_code(trainer.retrieval_loader, trainer.args.retrieval_num)
    q = qt[k_bits].to(trainer.device)
    r = ri[k_bits].to(trainer.device)
    sims = torch.matmul(q, r.t()).cpu().numpy()

    # Select interesting queries
    selected_indices = _select_interesting_indices(trainer.query_loader.dataset, max_queries, trainer.args.dataset)

    for qidx in selected_indices:
        if qidx >= q.shape[0]: continue

        sub = os.path.join(out_dir, f't2i_bits{k_bits}_q{qidx}')
        Path(sub).mkdir(parents=True, exist_ok=True)
        qimg_t = trainer.query_loader.dataset._load_image(int(qidx))
        qimg = _to_pil_denorm(qimg_t)
        qimg.save(os.path.join(sub, 'query.jpg'))
        _pseudocolor(qimg).save(os.path.join(sub, 'query_pseudo.jpg'))
        q_text = _first_caption(trainer.query_loader.dataset, int(qidx))
        order = np.argsort(-sims[qidx])[:topk]
        lines = []
        for i, ridx in enumerate(order):
            img_t = trainer.retrieval_loader.dataset._load_image(int(ridx))
            base = _to_pil_denorm(img_t)
            base.save(os.path.join(sub, f'top{i+1}.jpg'))
            simv = float(sims[qidx, ridx])
            badge = _annotate_badge(base, str(i+1), f'sim={simv:.2f}')
            badge.save(os.path.join(sub, f'top{i+1}_color.jpg'))
            _pseudocolor(base).save(os.path.join(sub, f'top{i+1}_pseudo.jpg'))
            cap = _first_caption(trainer.retrieval_loader.dataset, int(ridx))
            idx_path = str(trainer.retrieval_loader.dataset.indexs[int(ridx)])
            lines.append(f'{i+1}. sim={simv:.4f} | {cap} | {idx_path}')
            with open(os.path.join(sub, f'top{i+1}.txt'), 'w', encoding='utf-8') as tf:
                tf.write(cap)
        with open(os.path.join(sub, 'info.txt'), 'w', encoding='utf-8') as f:
            f.write(f'dataset={trainer.args.dataset} tag={trainer.args.exp_tag} bits={k_bits}\n')
            f.write(f'query_text: {q_text}\n')
            f.write('\n'.join(lines))

def _save_i2t_raw(trainer, k_bits, topk, out_dir, max_queries=6):
    qi, qt = trainer.get_code(trainer.query_loader, trainer.args.query_num)
    ri, rt = trainer.get_code(trainer.retrieval_loader, trainer.args.retrieval_num)
    q = qi[k_bits].to(trainer.device)
    r = rt[k_bits].to(trainer.device)
    sims = torch.matmul(q, r.t()).cpu().numpy()

    # Select interesting queries
    selected_indices = _select_interesting_indices(trainer.query_loader.dataset, max_queries, trainer.args.dataset)

    for qidx in selected_indices:
        if qidx >= q.shape[0]: continue

        sub = os.path.join(out_dir, f'i2t_bits{k_bits}_q{qidx}')
        Path(sub).mkdir(parents=True, exist_ok=True)
        qimg_t = trainer.query_loader.dataset._load_image(int(qidx))
        qimg = _to_pil_denorm(qimg_t)
        qimg.save(os.path.join(sub, 'query.jpg'))
        q_text = _first_caption(trainer.query_loader.dataset, int(qidx))
        order = np.argsort(-sims[qidx])[:topk]
        lines = []
        for i, ridx in enumerate(order):
            simv = float(sims[qidx, ridx])
            cap = _first_caption(trainer.retrieval_loader.dataset, int(ridx))
            lines.append(f'{i+1}. sim={simv:.4f} | {cap}')
            with open(os.path.join(sub, f'top{i+1}.txt'), 'w', encoding='utf-8') as tf:
                tf.write(cap)
        with open(os.path.join(sub, 'info.txt'), 'w', encoding='utf-8') as f:
            f.write(f'dataset={trainer.args.dataset} tag={trainer.args.exp_tag} bits={k_bits}\n')
            f.write(f'query_text: {q_text}\n')
            f.write('\n'.join(lines))

def main():
    args = get_args()
    args.is_train = False
    if args.pretrained == "":
        raise RuntimeError("please set --pretrained to a trained model path")
    args.caption_file = "caption.txt"
    args.index_file = "index.mat"
    args.label_file = "label.mat"
    trainer = TrainerAsym(args)
    k_list = list(map(int, args.k_bits_list.split(',')))
    out_dir = os.path.join(args.save_dir, 'examples')
    for k_bits in k_list:
        _save_t2i_raw(trainer, k_bits=k_bits, topk=5, out_dir=out_dir, max_queries=6)
        _save_i2t_raw(trainer, k_bits=k_bits, topk=5, out_dir=out_dir, max_queries=6)

if __name__ == '__main__':
    main()
