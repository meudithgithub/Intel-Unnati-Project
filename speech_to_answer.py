# speech_to_answer.py

import whisper
import re
from transformers import pipeline

# Load models globally
asr_model = whisper.load_model("small")
extractive_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
generative_qa = pipeline("text2text-generation", model="google/flan-t5-large")

# Keywords that indicate a need for explanation (generative)
explanation_keywords = [
    "explain", "meaning", "i didn't understand", "what does", "why",
    "elaborate", "importance", "significance", "define"
]

def is_explanation_query(question):
    """Heuristic to determine if the question needs generative QA"""
    question_lower = question.lower()
    return any(re.search(rf"\b{kw}\b", question_lower) for kw in explanation_keywords)

def transcribe_audio(audio_path):
    """Transcribes audio file to text"""
    result = asr_model.transcribe(audio_path)
    return result["text"]

def answer_question(question, context):
    """Routes to either extractive or generative QA model based on question intent"""
    if is_explanation_query(question):
        prompt = f"You are an educational assistant. Read the following context and answer the question in detail:\nContext: {context}\nQuestion: {question}"
        response = generative_qa(prompt, max_length=100, do_sample=False)
        return response[0]['generated_text']
    else:
        response = extractive_qa(question=question, context=context)
        return response['answer']

def process_audio_qa(audio_path, context):
    """Full pipeline from audio to answer"""
    print("\n🎧 Transcribing audio...")
    question = transcribe_audio(audio_path)
    print(f"🧠 Transcribed Question: {question}")

    print("🔍 Answering...")
    answer = answer_question(question, context)
    print(f"✅ Answer: {answer}")
    return question, answer
