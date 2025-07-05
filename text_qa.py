import re
import time
import logging
from typing import Dict, Any, Optional
from transformers.pipelines import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance tracking
performance_log = {
    "extractive_inference_times": [],
    "generative_inference_times": [],
    "total_queries": 0,
    "openvino_available": False,
    "model_backend": "CPU"
}

# OpenVINO imports and setup
try:
    from optimum.intel import OVQuestionAnsweringPipeline, OVText2TextGenerationPipeline
    from openvino.runtime import Core
    import openvino as ov
    
    # Initialize OpenVINO core
    core = Core()
    available_devices = core.available_devices
    logger.info(f"✅ OpenVINO available. Devices: {available_devices}")
    
    # Set default device
    default_device = "CPU"
    if "CPU" in available_devices:
        logger.info("🎯 Using CPU device for OpenVINO")
    else:
        logger.warning("⚠️ CPU device not available, using first available device")
        default_device = available_devices[0] if available_devices else "CPU"
    
    performance_log["openvino_available"] = True
    performance_log["model_backend"] = "OpenVINO"
    
except ImportError as e:
    logger.warning(f"⚠️ OpenVINO not available: {e}")
    performance_log["openvino_available"] = False
    performance_log["model_backend"] = "CPU"

# Explanation keywords
explanation_keywords = [
    "explain", "describe", "characteristics", "features", "purpose", "goal",
    "what is", "tell me", "elaborate", "in detail", "importance", "meaning"
]

# Initialize models
extractive_qa = None
generative_qa = None

def initialize_models():
    """Initialize QA models with OpenVINO acceleration if available"""
    global extractive_qa, generative_qa
    
    try:
        if performance_log["openvino_available"]:
            logger.info("🚀 Initializing OpenVINO-accelerated models...")
            
            # Initialize OpenVINO models
            extractive_qa = OVQuestionAnsweringPipeline.from_pretrained(
                "deepset/roberta-base-squad2",
                device=default_device,
                compile=True
            )
            
            generative_qa = OVText2TextGenerationPipeline.from_pretrained(
                "google/flan-t5-large",
                device=default_device,
                compile=True
            )
            
            logger.info("✅ OpenVINO models initialized successfully")
            
        else:
            logger.info("📦 Initializing CPU models (OpenVINO not available)...")
            
            # Fallback to CPU models
            extractive_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
            generative_qa = pipeline("text2text-generation", model="google/flan-t5-large")
            
            logger.info("✅ CPU models initialized successfully")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        logger.info("🔄 Falling back to CPU models...")
        
        # Emergency fallback
        try:
            extractive_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
            generative_qa = pipeline("text2text-generation", model="google/flan-t5-large")
            performance_log["model_backend"] = "CPU (Fallback)"
            logger.info("✅ Fallback models initialized successfully")
        except Exception as fallback_error:
            logger.error(f"❌ Fallback initialization failed: {fallback_error}")
            raise

def is_explanation_query(question):
    """Check if question is more suited for generative answering."""
    question_lower = question.lower()
    return any(re.search(rf"\b{kw}\b", question_lower) for kw in explanation_keywords)

def predict_with_performance(pipeline_func, *args, **kwargs):
    """Execute prediction with performance measurement"""
    start_time = time.time()
    
    try:
        result = pipeline_func(*args, **kwargs)
        inference_time = time.time() - start_time
        
        # Log performance based on model type
        if "question-answering" in str(type(pipeline_func)).lower():
            performance_log["extractive_inference_times"].append(inference_time)
        else:
            performance_log["generative_inference_times"].append(inference_time)
        
        performance_log["total_queries"] += 1
        
        return result, inference_time
        
    except Exception as e:
        inference_time = time.time() - start_time
        logger.error(f"❌ Prediction failed: {e}")
        return None, inference_time

def answer_question(question, context):
    """Answer question using accelerated models with performance tracking"""
    
    # Ensure models are initialized
    if extractive_qa is None or generative_qa is None:
        initialize_models()
    
    try:
        if is_explanation_query(question):
            # Use generative QA for explanation queries
            prompt = f"You are an educational assistant. Read the following context and answer the question in detail:\n\nContext:\n{context}\n\nQuestion:\n{question}"
            
            result, inference_time = predict_with_performance(
                generative_qa, 
                prompt, 
                max_length=256, 
                do_sample=False
            )
            
            if result:
                logger.info(f"🎯 Generative QA: {inference_time:.4f}s")
                # Handle different result formats
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', str(result[0]))
                elif isinstance(result, dict):
                    return result.get('generated_text', str(result))
                else:
                    return str(result)
            else:
                return "Error: Failed to generate answer"
                
        else:
            # Use extractive QA for fact-based queries
            result, inference_time = predict_with_performance(
                extractive_qa,
                question=question,
                context=context
            )
            
            if result:
                logger.info(f"🎯 Extractive QA: {inference_time:.4f}s")
                # Handle different result formats
                if isinstance(result, dict):
                    return result.get('answer', str(result))
                else:
                    return str(result)
            else:
                return "Error: Failed to extract answer"
                
    except Exception as e:
        logger.error(f"❌ Question answering failed: {e}")
        return f"Error: {str(e)}"

def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics"""
    stats = performance_log.copy()
    
    # Calculate averages
    if stats["extractive_inference_times"]:
        stats["avg_extractive_time"] = sum(stats["extractive_inference_times"]) / len(stats["extractive_inference_times"])
        stats["min_extractive_time"] = min(stats["extractive_inference_times"])
        stats["max_extractive_time"] = max(stats["extractive_inference_times"])
    else:
        stats["avg_extractive_time"] = 0
        stats["min_extractive_time"] = 0
        stats["max_extractive_time"] = 0
    
    if stats["generative_inference_times"]:
        stats["avg_generative_time"] = sum(stats["generative_inference_times"]) / len(stats["generative_inference_times"])
        stats["min_generative_time"] = min(stats["generative_inference_times"])
        stats["max_generative_time"] = max(stats["generative_inference_times"])
    else:
        stats["avg_generative_time"] = 0
        stats["min_generative_time"] = 0
        stats["max_generative_time"] = 0
    
    return stats

def reset_performance_log():
    """Reset performance tracking"""
    global performance_log
    performance_log = {
        "extractive_inference_times": [],
        "generative_inference_times": [],
        "total_queries": 0,
        "openvino_available": performance_log.get("openvino_available", False),
        "model_backend": performance_log.get("model_backend", "CPU")
    }

# Initialize models on import
try:
    initialize_models()
except Exception as e:
    logger.error(f"❌ Failed to initialize models on import: {e}")
