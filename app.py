# app.py

import gradio as gr
import threading
import cv2
import time
import json
from datetime import datetime

from intent_router import detect_intent, detect_emotion, detect_emotion_from_frame
from text_qa import answer_question as text_qa, get_performance_stats, reset_performance_log
from speech_to_answer import process_audio_qa
from image_qa import answer_question_from_image

# Flag to stop the webcam thread gracefully
stop_emotion_thread = False

# Background thread for continuous emotion detection via webcam
def emotion_detection_loop():
    global stop_emotion_thread
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Could not open webcam for emotion detection.")
        return

    print("[INFO] Starting continuous emotion detection via webcam...")

    while not stop_emotion_thread:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to capture frame.")
            continue

        emotion, _ = detect_emotion_from_frame(frame)
        print(f"[Emotion Detection] Current emotion: {emotion}")

        time.sleep(1)  # Avoid CPU overuse

    cap.release()
    print("[INFO] Emotion detection stopped.")

# Performance monitoring functions
def get_performance_dashboard():
    """Get formatted performance statistics for the dashboard"""
    try:
        stats = get_performance_stats()
        
        # Format the data for display
        dashboard_data = {
            "🚀 OpenVINO Status": {
                "OpenVINO Available": "✅ Yes" if stats.get("openvino_available", False) else "❌ No",
                "Model Backend": stats.get("model_backend", "Unknown"),
                "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "📊 Performance Metrics": {
                "Total Queries": stats.get("total_queries", 0),
                "Extractive Queries": len(stats.get("extractive_inference_times", [])),
                "Generative Queries": len(stats.get("generative_inference_times", [])),
                "Model Backend": stats.get("model_backend", "Unknown")
            },
            "⚡ Extractive QA Performance": {
                "Average Time": f"{stats.get('avg_extractive_time', 0):.4f}s",
                "Min Time": f"{stats.get('min_extractive_time', 0):.4f}s",
                "Max Time": f"{stats.get('max_extractive_time', 0):.4f}s",
                "Total Queries": len(stats.get("extractive_inference_times", []))
            },
            "🎯 Generative QA Performance": {
                "Average Time": f"{stats.get('avg_generative_time', 0):.4f}s",
                "Min Time": f"{stats.get('min_generative_time', 0):.4f}s",
                "Max Time": f"{stats.get('max_generative_time', 0):.4f}s",
                "Total Queries": len(stats.get("generative_inference_times", []))
            }
        }
        
        return json.dumps(dashboard_data, indent=2)
    except Exception as e:
        return f"Error retrieving performance data: {str(e)}"

def reset_performance_metrics():
    """Reset all performance tracking metrics"""
    try:
        reset_performance_log()
        return "✅ Performance metrics have been reset successfully!"
    except Exception as e:
        return f"❌ Error resetting performance metrics: {str(e)}"

# Gradio assistant logic
def ai_assistant(context, text_input, audio_file, image_file, image_question):
    result = {}

    # Intent-based QnA
    if text_input and context:
        result["Answer"] = text_qa(text_input, context)

    elif audio_file and context:
        question, answer = process_audio_qa(audio_file, context)
        result["Question (from audio)"] = question
        result["Answer"] = answer

    elif image_file and image_question and context:
        out = answer_question_from_image(image_file, image_question)
        result["Caption"] = out["caption"]
        result["Answer"] = out["answer"]
        result["Mode"] = out["mode"]

    else:
        result["Error"] = "Please provide context and at least one input: text, audio, or image + question."

    return result

# Create Gradio interface with tabs
with gr.Blocks(title="AI-Powered Classroom Assistant with Performance Monitoring") as iface:
    gr.Markdown("# 🧠 AI-Powered Classroom Assistant")
    gr.Markdown("### Answer questions from text, speech, or image using context. Detect student emotion using uploaded photo.")
    
    with gr.Tabs():
        # Main Assistant Tab
        with gr.TabItem("🤖 AI Assistant"):
            with gr.Row():
                with gr.Column():
                    context_input = gr.Textbox(label="📘 Context (Paste your text/lecture content here)", lines=6)
                    text_input = gr.Textbox(label="📝 Text Question (if any)")
                    audio_file = gr.Audio(type="filepath", label="🎤 Upload Audio (optional)")
                    image_file = gr.Image(type="filepath", label="🖼️ Upload Image (optional)")
                    image_question = gr.Textbox(label="🖼️ Question about Image (optional)")
                    
                    submit_btn = gr.Button("🚀 Process Request", variant="primary")
                
                with gr.Column():
                    output_json = gr.JSON(label="📋 Results")
        
        # Performance Monitoring Tab
        with gr.TabItem("📊 Performance Dashboard"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🚀 OpenVINO Text QA Performance")
                    gr.Markdown("Monitor real-time performance metrics for text question answering with OpenVINO acceleration.")
                    
                    refresh_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")
                    reset_btn = gr.Button("🔄 Reset Metrics", variant="secondary")
                    
                with gr.Column():
                    performance_output = gr.JSON(label="📈 Performance Statistics")
            
            with gr.Row():
                gr.Markdown("""
                ### 📋 Performance Metrics Explained:
                
                **🚀 OpenVINO Status:**
                - **OpenVINO Available**: Whether OpenVINO acceleration is active
                - **Model Backend**: Current backend (OpenVINO/CPU)
                - **Last Updated**: Timestamp of last metric update
                
                **📊 Performance Metrics:**
                - **Total Queries**: Total number of text QA queries processed
                - **Extractive Queries**: Number of fact-based queries (who, what, when, where)
                - **Generative Queries**: Number of explanation queries (explain, describe, why)
                - **Model Backend**: Current backend being used
                
                **⚡ Extractive QA Performance:**
                - **Average/Min/Max Time**: Performance statistics for fact-based queries
                - **Total Queries**: Number of extractive queries processed
                
                **🎯 Generative QA Performance:**
                - **Average/Min/Max Time**: Performance statistics for explanation queries
                - **Total Queries**: Number of generative queries processed
                
                **💡 Performance Tips:**
                - OpenVINO typically provides 20-50% faster inference times
                - Extractive QA is generally faster than Generative QA
                - Performance improves with repeated queries due to model caching
                """)
    
    # Event handlers
    submit_btn.click(
        fn=ai_assistant,
        inputs=[context_input, text_input, audio_file, image_file, image_question],
        outputs=output_json
    )
    
    refresh_btn.click(
        fn=get_performance_dashboard,
        inputs=[],
        outputs=performance_output
    )
    
    reset_btn.click(
        fn=reset_performance_metrics,
        inputs=[],
        outputs=performance_output
    )
    
    # Auto-refresh performance metrics every 5 seconds
    iface.load(
        fn=get_performance_dashboard,
        inputs=[],
        outputs=performance_output
    )

if __name__ == "__main__":
    # Start background emotion detection thread (optional)
    emotion_thread = threading.Thread(target=emotion_detection_loop, daemon=True)
    emotion_thread.start()

    try:
        iface.launch()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    finally:
        stop_emotion_thread = True
        # Don't join daemon thread - it will be terminated automatically
