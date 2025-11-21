

from PIL import Image
import os
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
import torch
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from random import uniform
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk
nltk.download('punkt_tab')

# Download nltk data if not available
nltk.download("punkt", quiet=True)

# -------------------------------
# 1️⃣ Load BLIP + Swin Transformer Model
# -------------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# -------------------------------
# 2️⃣ Caption Generation Function
# -------------------------------
def generate_caption(img, num_beams=3, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
    inputs = processor(img, return_tensors="pt")
    output = model.generate(
        **inputs,
        num_beams=int(num_beams),
        max_length=int(max_length),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p)
    )
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# -------------------------------
# 3️⃣ Fuzzy Logic Evaluation System
# -------------------------------
bleu = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'BLEU')
similarity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Similarity')
quality = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'Quality')

bleu.automf(3)
similarity.automf(3)
quality.automf(3)

rule1 = ctrl.Rule(bleu['good'] & similarity['good'], quality['good'])
rule2 = ctrl.Rule(bleu['average'] | similarity['average'], quality['average'])
rule3 = ctrl.Rule(bleu['poor'] & similarity['poor'], quality['poor'])

quality_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
quality_eval = ctrl.ControlSystemSimulation(quality_ctrl)

def fuzzy_quality(bleu_score, similarity_score):
    quality_eval.input['BLEU'] = bleu_score
    quality_eval.input['Similarity'] = similarity_score
    quality_eval.compute()
    return quality_eval.output['Quality']

# -------------------------------
# 4️⃣ Real Evaluation Metrics (BLEU + BERTScore)
# -------------------------------
def evaluate_caption(reference_caption, generated_caption):
    # BLEU (syntactic similarity)
    ref_tokens = nltk.word_tokenize(reference_caption.lower())
    gen_tokens = nltk.word_tokenize(generated_caption.lower())
    smoothie = SmoothingFunction().method4
    bleu_score_val = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)

    # BERTScore (semantic similarity)
    P, R, F1 = bert_score([generated_caption], [reference_caption], lang="en", rescale_with_baseline=True)
    similarity_score = float(F1[0])

    return bleu_score_val, similarity_score

# -------------------------------
# 5️⃣ Dragonfly Optimization Algorithm
# -------------------------------
def dragonfly_optimization(img, reference_caption, iterations=5):
    lower_bounds = [2, 30, 0.7, 20, 0.7]
    upper_bounds = [6, 80, 1.5, 80, 0.95]
    best_params = None
    best_score = -1

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
        bleu_score_val, sim_score = evaluate_caption(reference_caption, caption)
        q = fuzzy_quality(bleu_score_val, sim_score)

        if q > best_score:
            best_score = q
            best_params = params

    return best_params, best_score

# -------------------------------
# 6️⃣ Sequence Captioning for Folder
# -------------------------------
def caption_sequence(folder_path, reference_caption="A general description of the image."):
    captions = []
    for img_name in sorted(os.listdir(folder_path)):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(os.path.join(folder_path, img_name))
            best_params, _ = dragonfly_optimization(img, reference_caption)
            caption = generate_caption(
                img,
                num_beams=best_params[0],
                max_length=best_params[1],
                temperature=best_params[2],
                top_k=best_params[3],
                top_p=best_params[4],
            )
            captions.append((img_name, caption))
    return captions

# -------------------------------
# 7️⃣ Gradio Interface
# -------------------------------
def run_single_image(img, reference_caption):
    best_params, best_score = dragonfly_optimization(img, reference_caption)
    caption = generate_caption(
        img,
        num_beams=best_params[0],
        max_length=best_params[1],
        temperature=best_params[2],
        top_k=best_params[3],
        top_p=best_params[4],
    )
    bleu, sim = evaluate_caption(reference_caption, caption)
    return f"""
    🖼️ Caption: {caption}
    \n🔧 Optimized Params: {np.round(best_params, 2)}
    \n📊 BLEU Score: {round(bleu, 3)}
    \n🤖 BERT Similarity: {round(sim, 3)}
    \n🧠 Fuzzy Quality: {round(best_score, 3)}
    """

def run_folder(folder, reference_caption):
    results = caption_sequence(folder, reference_caption)
    text = "\n".join([f"{img}: {cap}" for img, cap in results])
    return text

demo = gr.Interface(
    fn=run_single_image,
    inputs=[
        gr.Image(label="Upload Image"),
        gr.Textbox(label="Reference Caption (for evaluation)", placeholder="Describe what the image is about...")
    ],
    outputs=[gr.Textbox(label="Generated Caption with Evaluation")],
    title="🧠 BLIP + Swin Transformer Captioning with Fuzzy & Dragonfly Optimization",
    description="Upload an image and a reference caption to generate optimized captions using BLEU + BERTScore evaluation."
)

demo2 = gr.Interface(
    fn=run_folder,
    inputs=[
        gr.File(label="Upload Folder of Images"),
        gr.Textbox(label="Reference Caption for Sequence", placeholder="General description of images...")
    ],
    outputs=[gr.Textbox(label="Sequence Captions")],
    title="📷 Sequential Image Captioning",
    description="Upload a folder of images to generate optimized captions for each frame in sequence."
)

app = gr.TabbedInterface([demo, demo2], ["Single Image", "Image Sequence"])
app.launch()
