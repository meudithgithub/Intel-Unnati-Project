# image_qa.py

from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import re

# Load models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

extractive_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
generative_qa = pipeline("text2text-generation", model="google/flan-t5-base")

# Explanation-type keywords
explanation_keywords = [
    "explain", "why", "how", "meaning", "i didn't understand", "describe", "elaborate", "happening"
]

def is_explanation_query(question):
    question_lower = question.lower()
    return any(re.search(rf"\b{kw}\b", question_lower) for kw in explanation_keywords)

def answer_question_from_image(image_path, question):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Error loading image: {e}"}

    # Generate caption from image
    inputs = caption_processor(images=image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)

    # Choose QA method
    if is_explanation_query(question):
        prompt = f"Context: {caption}\nQuestion: {question}"
        response = generative_qa(prompt, max_length=100, do_sample=False)[0]['generated_text']
        mode = "generative"
    else:
        response = extractive_qa(question=question, context=caption)['answer']
        mode = "extractive"

    return {
        "caption": caption,
        "question": question,
        "answer": response,
        "mode": mode
    }
    