import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Default system prompt for GPU recommendations
DEFAULT_SYSTEM_PROMPT = """
You are an expert cloud infrastructure consultant specializing in GPU selection for various workloads.
Your task is to analyze GPU options and provide tailored recommendations based on:
1. The workload requirements (e.g., ML training, inference, rendering)
2. Performance needs (VRAM, compute capacity, etc.)
3. Budget constraints
4. Regional preferences

Provide a clear, detailed recommendation with rationale.
"""

def format_gpu_for_prompt(gpu_data, index=None):
    """Format a GPU instance for inclusion in a prompt"""
    prefix = f"Option {index}: " if index is not None else ""
    return f"""{prefix}{gpu_data.get('resource_name', gpu_data.get('gpu_description', 'Unknown GPU'))}
- vCPUs: {gpu_data.get('vcpus', 'N/A')}
- RAM: {gpu_data.get('ram', 'N/A')} GB
- GPU: {gpu_data.get('gpu_description', 'N/A')}
- Region: {gpu_data.get('region', 'N/A')}
- Price/hour: ${gpu_data.get('price_per_hour', 'N/A')}
- Price/month: ${gpu_data.get('price_per_month', 'N/A')}"""

def query_llm_for_recommendation(query, gpu_data, model="gpt-3.5-turbo", format="json"):
    """
    Query LLM to get GPU recommendations based on filtered data
    
    Args:
        query: User's natural language query
        gpu_data: List of dictionaries containing GPU information
        model: OpenAI model to use
        format: Response format ('text' or 'json')
        
    Returns:
        LLM response (JSON or text)
    """
    # Prepare context with GPU options
    gpu_options = []
    for i, gpu in enumerate(gpu_data[:10], 1):  # Limit to 10 options for prompt size
        gpu_options.append(format_gpu_for_prompt(gpu, i))
    
    gpu_context = "\n\n".join(gpu_options)
    
    # Construct user prompt
    user_prompt = f"""
User query: {query}

Available GPU options:
{gpu_context}

Based on the user's query and the available GPU options, provide a recommendation.
Explain your reasoning and highlight the best fit for their specific needs.
"""
    
    # Set up system prompt based on requested format
    if format == "json":
        system_prompt = DEFAULT_SYSTEM_PROMPT + """
Format your response as JSON with the following structure:
{
  "recommendation": {
    "primary": {
      "name": "GPU name/description",
      "option_number": 1,
      "rationale": "Why this GPU is best"
    },
    "alternatives": [
      {
        "name": "Alternative GPU",
        "option_number": 2,
        "rationale": "When to consider this instead"
      }
    ]
  },
  "analysis": {
    "workload_match": "Explanation of how this matches their workload",
    "cost_efficiency": "Cost-benefit analysis",
    "limitations": "Any limitations to be aware of"
  },
  "explanation": "Human-readable paragraph explaining the recommendation"
}
"""
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    # Make API call
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 1000
        }
        
        # Add response format for JSON
        if format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        
        response = client.chat.completions.create(**kwargs)
        
        result = response.choices[0].message.content
        
        # Parse JSON if that's the requested format
        if format == "json":
            try:
                return json.loads(result)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return {"error": "Failed to parse recommendation", "raw_response": result}
        
        return result
        
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return {"error": f"Failed to generate recommendation: {str(e)}"}

def simplify_recommendation(gpu_recommendation):
    """Convert a complex recommendation to simpler format"""
    if isinstance(gpu_recommendation, str):
        return {"explanation": gpu_recommendation}
        
    if "error" in gpu_recommendation:
        return {"error": gpu_recommendation["error"]}
        
    try:
        if "recommendation" in gpu_recommendation:
            primary = gpu_recommendation["recommendation"].get("primary", {})
            explanation = gpu_recommendation.get("explanation", "")
            
            return {
                "recommended_gpu": primary.get("name"),
                "rationale": primary.get("rationale"),
                "explanation": explanation
            }
        return {"explanation": str(gpu_recommendation)}
    except Exception as e:
        logger.error(f"Error simplifying recommendation: {e}")
        return {"explanation": "Could not process recommendation details."}

if __name__ == "__main__":
    # Example usage
    sample_query = "I need a GPU for training large language models with at least 40GB memory"
    sample_data = [
        {
            "resource_name": "a2-highgpu-1g",
            "gpu_description": "NVIDIA A100 80GB",
            "vcpus": 12,
            "ram": 85,
            "region": "us-central1",
            "price_per_hour": 3.67,
            "price_per_month": 2678.0
        },
        {
            "resource_name": "g5-standard-4",
            "gpu_description": "NVIDIA A10G",
            "vcpus": 16,
            "ram": 64,
            "region": "us-east-1",
            "price_per_hour": 1.21,
            "price_per_month": 883.0
        }
    ]
    
    # Test a recommendation
    result = query_llm_for_recommendation(sample_query, sample_data)
    print(json.dumps(result, indent=2))