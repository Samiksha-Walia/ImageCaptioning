import os
import re
import zipfile
import tempfile
import numpy as np
import nltk
nltk.download("punkt", quiet=True)

from PIL import Image
from random import uniform
from functools import lru_cache

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

import skfuzzy.control as ctrl
import gradio as gr
import pandas as pd

# ---------------------------
# 1) Load model & processor
# ---------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=False)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# ---------------------------
# 2) Helpers
# ---------------------------
def clean_for_bleu(text):
    return nltk.word_tokenize(text.lower())

def clean_for_display(text):
    return text


# -----------------------------------
# 4) Cached BERTScore (fix included)
# -----------------------------------
@lru_cache(maxsize=512)
def cached_bert_score(gen, refs_tuple):
    refs = list(refs_tuple)
    cand_list = [gen] * len(refs)
    P, R, F1 = bert_score(cand_list, refs, lang="en")
    return float(F1.max().item())   # fixed — tensor safe


# ---------------------------
# 3) Caption generation
# ---------------------------
def generate_caption(img, num_beams=3, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
    num_beams = int(max(1, round(num_beams)))
    inputs = processor(img, return_tensors="pt")
    output = model.generate(
        **inputs,
        num_beams=num_beams,
        max_length=int(max_length),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    )
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


# ---------------------------
# 5) Compute metrics
# ---------------------------
def compute_metrics(reference, generated):
    smoothie = SmoothingFunction().method7

    if isinstance(reference, list):
        refs = [clean_for_bleu(r) for r in reference]
    else:
        refs = [clean_for_bleu(reference)]

    gen_tokens = clean_for_bleu(generated)

    bleu1 = sentence_bleu(refs, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(refs, gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = sentence_bleu(refs, gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(refs, gen_tokens, smoothing_function=smoothie)

    if isinstance(reference, list):
        refs_tuple = tuple(reference)
    else:
        refs_tuple = (reference,)

    bert_f1 = cached_bert_score(generated, refs_tuple)

    return bleu1, bleu2, bleu3, bleu4, bert_f1


# ---------------------------
# 6) Fuzzy CQS
# ---------------------------
bleu1_fz = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'BLEU1')
bleu4_fz = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'BLEU4')
bert_fz  = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'BERT')
quality  = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'Quality')

names = ['very_poor', 'poor', 'average', 'good', 'excellent']
bleu1_fz.automf(5, names=names)
bleu4_fz.automf(5, names=names)
bert_fz.automf(5, names=names)
quality.automf(5, names=names)

rules = [
    ctrl.Rule(bert_fz['excellent'] & bleu4_fz['good'], quality['excellent']),
    ctrl.Rule(bert_fz['excellent'] & bleu1_fz['average'], quality['good']),
    ctrl.Rule(bleu4_fz['excellent'], quality['excellent']),
    ctrl.Rule(bert_fz['good'] & bleu1_fz['good'], quality['good']),
    ctrl.Rule(bert_fz['average'] | bleu1_fz['very_poor'], quality['poor']),
    ctrl.Rule(bert_fz['very_poor'] & bleu1_fz['very_poor'], quality['very_poor']),
]

quality_ctrl = ctrl.ControlSystem(rules)
quality_eval = ctrl.ControlSystemSimulation(quality_ctrl)

def fuzzy_cqs(bleu1, bleu4, bert):
    quality_eval.input['BLEU1'] = float(bleu1)
    quality_eval.input['BLEU4'] = float(bleu4)
    quality_eval.input['BERT']  = float(bert)
    quality_eval.compute()
    return float(quality_eval.output['Quality'])


# ---------------------------
# 7) Dragonfly optimization
# ---------------------------
def dragonfly_optimization(img, reference_caption, iterations=12):
    lower_bounds = [2, 35, 0.7, 20, 0.8]
    upper_bounds = [6, 55, 1.1, 40, 0.95]

    best_params = None
    best_score = -1.0

    

    for _ in range(iterations):
        params = [uniform(l, u) for l, u in zip(lower_bounds, upper_bounds)]

        caption = generate_caption(
            img,
            num_beams=params[0],
            max_length=params[1],
            temperature=params[2],
            top_k=params[3],
            top_p=params[4],
        )

        bleu1, bleu2, bleu3, bleu4, bert_f1 = compute_metrics(reference_caption, caption)
        q = fuzzy_cqs(bleu1, bleu4, bert_f1)

        if q > best_score:
            best_score = q
            best_params = params

        if best_score >= 0.995:
            break

    return best_params, best_score


# ---------------------------
# 8) ZIP ACCEPTANCE + sequence captioning
# ---------------------------
def caption_sequence(folder_path, reference_caption="A general description.", iterations=12):
    results = []

    # Walk through all subfolders
    for root, _, files in os.walk(folder_path):
        for img_name in files:
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, img_name)
                img = Image.open(img_path).convert("RGB")

                baseline_caption = generate_caption(img)
                best_params, best_score = dragonfly_optimization(img, reference_caption, iterations)

                optimized_caption = generate_caption(
                    img,
                    num_beams=best_params[0],
                    max_length=best_params[1],
                    temperature=best_params[2],
                    top_k=best_params[3],
                    top_p=best_params[4],
                )

                b1, b2, b3, b4, bbert = compute_metrics(reference_caption, baseline_caption)
                bCQS = fuzzy_cqs(b1, b4, bbert)

                o1, o2, o3, o4, obert = compute_metrics(reference_caption, optimized_caption)
                oCQS = fuzzy_cqs(o1, o4, obert)

                results.append({
                    "image": img_name,
                    "baseline_caption": baseline_caption,
                    "optimized_caption": optimized_caption,
                    "BLEU1_base": b1, "BLEU2_base": b2, "BLEU3_base": b3, "BLEU4_base": b4,
                    "BERT_base": bbert, "CQS_base": bCQS,
                    "BLEU1_opt": o1, "BLEU2_opt": o2, "BLEU3_opt": o3, "BLEU4_opt": o4,
                    "BERT_opt": obert, "CQS_opt": oCQS,
                    "best_params": best_params
                })

    return results


# ---------------------------
# ZIP extraction helper
# ---------------------------
def extract_zip_to_temp(zip_file_path):
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    return temp_dir



import ollama

def build_story(captions, model="mistral"):
    """
    Converts a list of image captions into a connected story using a local Ollama model.
    """
    # Build prompt
    prompt = "These are sequential image captions. Convert them into a connected, smooth story:\n\n"
    
    for i, cap in enumerate(captions, 1):
        prompt += f"{i}. {cap}\n"
    
    prompt += "\nWrite a single coherent story connecting all events."

    # Call Ollama
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract the generated text
    story = response.get("message", {}).get("content", "")

    return story


# ---------------------------
# 9) Gradio — single image
# ---------------------------
def run_single_image(img, reference_caption, iterations=12):

    if isinstance(reference_caption, str) and "\n" in reference_caption:
        refs = [r.strip() for r in reference_caption.split("\n") if r.strip()]
        reference = refs if len(refs) > 0 else reference_caption
    else:
        reference = reference_caption

    baseline_caption = generate_caption(img)

    best_params, best_score = dragonfly_optimization(img, baseline_caption, iterations)
    opt_caption = generate_caption(
        img,
        num_beams=best_params[0],
        max_length=best_params[1],
        temperature=best_params[2],
        top_k=best_params[3],
        top_p=best_params[4],
    )

    b1, b2, b3, b4, bbert = compute_metrics(reference, baseline_caption)
    bCQS = fuzzy_cqs(b1, b4, bbert)

    o1, o2, o3, o4, obert = compute_metrics(reference, opt_caption)
    oCQS = fuzzy_cqs(o1, o4, obert)

    # ---- CSV export ----
    df = pd.DataFrame([{
        "baseline_caption": baseline_caption,
        "optimized_caption": opt_caption,
        "BLEU1_base": b1, "BLEU2_base": b2, "BLEU3_base": b3, "BLEU4_base": b4,
        "BERT_base": bbert, "CQS_base": bCQS,
        "BLEU1_opt": o1, "BLEU2_opt": o2, "BLEU3_opt": o3, "BLEU4_opt": o4,
        "BERT_opt": obert, "CQS_opt": oCQS,
        "best_params": best_params
    }])

    csv_path = os.path.join(tempfile.gettempdir(), "folder_results.csv")
    df.to_csv(csv_path, index=False)

    out_text = f"""
🖼️ Baseline Caption:
{baseline_caption}

🖼️ Optimized Caption:
{opt_caption}

--- Baseline Scores ---
BLEU-1: {b1:.4f}
BLEU-2: {b2:.4f}
BLEU-3: {b3:.4f}
BLEU-4: {b4:.4f}
BERTScore (F1): {bbert:.4f}
Fuzzy CQS: {bCQS:.4f}

--- Optimized Scores ---
BLEU-1: {o1:.4f}
BLEU-2: {o2:.4f}
BLEU-3: {o3:.4f}
BLEU-4: {o4:.4f}
BERTScore (F1): {obert:.4f}
Fuzzy CQS: {oCQS:.4f}

🔧 Best Params: {np.round(best_params, 3)}
    """

    return out_text, csv_path


# ---------------------------
# 10) Gradio — folder OR ZIP
# ---------------------------
def run_folder_upload(zip_or_folder, reference_caption, iterations=12):

    # Handle multi-ref
    if isinstance(reference_caption, str) and "\n" in reference_caption:
        refs = [r.strip() for r in reference_caption.split("\n") if r.strip()]
        reference = refs if len(refs) > 0 else reference_caption
    else:
        reference = reference_caption

    # Extract ZIP
    if zip_or_folder.name.endswith(".zip"):
        folder_path = extract_zip_to_temp(zip_or_folder.name)
    else:
        folder_path = zip_or_folder.name

    # Run sequence generation
    results = caption_sequence(folder_path, reference_caption=reference, iterations=iterations)

    # Create CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(folder_path, "folder_results.csv")
    df.to_csv(csv_path, index=False)

    # Format text output
    text_output = ""
    for row in results:
        text_output += f"""
📌 **{row['image']}**



Baseline Caption:
{row['baseline_caption']}

Optimized Caption:
{row['optimized_caption']}

--- Baseline ---
BLEU1={row['BLEU1_base']:.4f}, BLEU4={row['BLEU4_base']:.4f}, 
BERT={row['BERT_base']:.4f}, CQS={row['CQS_base']:.4f}

--- Optimized ---
BLEU1={row['BLEU1_opt']:.4f}, BLEU4={row['BLEU4_opt']:.4f}, 
BERT={row['BERT_opt']:.4f}, CQS={row['CQS_opt']:.4f}

Best params: {np.round(row['best_params'], 3)}

-----------------------------------
"""
        
    # -------------------------------
    # STORY GENERATION USING OLLAMA
    # -------------------------------
    optimized_captions = [row["optimized_caption"] for row in results]
    story = build_story(optimized_captions, model="mistral")

    text_output += "\n\n📖 **Generated Story**\n"
    text_output += story

    return text_output, csv_path



# ---------------------------
# 11) Gradio UI
# ---------------------------
single_inputs = [
    gr.Image(label="Upload Image"),
    gr.Textbox(label="Reference Caption", placeholder="Single or multiple refs (newline separated)"),
    gr.Slider(minimum=1, maximum=30, value=12, step=1, label="Optimizer iterations")
]

single_output = [
    gr.Markdown(label="Detailed Results"),
    gr.File(label="Download CSV"),
]

demo = gr.Interface(
    fn=run_single_image,
    inputs=single_inputs,
    outputs=single_output,
    title="🧠 BLIP + Fuzzy CQS + Dragonfly Optimization",
    description="Caption image with BLEU, BERTScore, Fuzzy CQS + CSV export"
)

folder_inputs = [
    gr.File(label="Upload ZIP or Folder"),
    gr.Textbox(label="Reference Caption", placeholder="Single or multiple refs (newline separated)"),
    gr.Slider(minimum=1, maximum=30, value=12, step=1, label="Optimizer iterations")
]

folder_output = [
    gr.Markdown(label="Sequence Captions"),
    gr.File(label="Download CSV"),
]



demo2 = gr.Interface(
    fn=run_folder_upload,
    inputs=folder_inputs,
    outputs=folder_output,
    title="📦 ZIP Folder Captioning",
    description="Upload ZIP of images → captions + CSV"
)

app = gr.TabbedInterface([demo, demo2], ["Single Image", "Folder / ZIP Sequence"],
                         css="""
    .boxed-md {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        padding: 20px !important;
        border-radius: 10px !important;
        border: 1px solid #444 !important;
        max-height: 600px !important;

        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    """)
app.launch()
