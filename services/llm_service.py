import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt template for GPU recommendations
SYSTEM_PROMPT = """
You are an expert cloud infrastructure consultant specializing in GPU selection for various workloads.
Your task is to analyze GPU options and provide tailored recommendations based on:
1. The workload requirements (e.g., ML training, inference, rendering)
2. Performance needs (VRAM, compute capacity, etc.)
3. Budget constraints
4. Regional preferences

Provide a clear, detailed recommendation with rationale. Include:
- Primary GPU recommendation with justification
- Alternative options if applicable
- Cost analysis (hourly, monthly)
- Performance considerations
- Any trade-offs the user should be aware of

Format your response as JSON with the following structure:
{
  "recommendation": {
    "primary": {
      "name": "GPU name/description",
      "rationale": "Why this GPU is best",
      "key_specs": ["Key specification 1", "Key specification 2"]
    },
    "alternatives": [
      {
        "name": "Alternative GPU",
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

def recommend_gpu_with_llm(query, gpu_data, user_preferences=None):
    """
    Generate GPU recommendations using LLM based on query and available GPU data
    
    Args:
        query: User's question or requirements
        gpu_data: List of available GPUs with specifications
        user_preferences: Optional dict of user preferences
        
    Returns:
        JSON response with recommendations
    """
    if not gpu_data:
        return {"error": "No GPU data available to analyze"}
    
    # Prepare context
    context = []
    for i, gpu in enumerate(gpu_data[:10]):  # Limit to top 10 GPUs for context
        # Use resource_name as the primary identifier
        gpu_name = gpu.get('resource_name', 'Unknown GPU')
        
        specs = [
            f"vCPUs: {gpu.get('vcpus', 'N/A')}",
            f"RAM: {gpu.get('ram', 'N/A')} GB",
            f"Country/Region: {gpu.get('country', gpu.get('region', 'N/A'))}",
            f"Operating System: {gpu.get('operating_system', 'N/A')}",
            f"Price/hour: ${gpu.get('price_per_hour', 'N/A')}",
            f"Price/month: ${gpu.get('price_per_month', 'N/A')}",
            f"Spot price: ${gpu.get('price_per_spot', 'N/A')}"
        ]
        context.append(f"GPU {i+1}: {gpu_name}\n" + "\n".join(specs))
    
    gpu_context = "\n\n".join(context)
    
    # Include user preferences if available
    preferences_text = ""
    if user_preferences:
        pref_items = []
        if user_preferences.get("preferred_region"):
            pref_items.append(f"Preferred region: {user_preferences['preferred_region']}")
        if user_preferences.get("workload_type"):
            pref_items.append(f"Typical workload: {user_preferences['workload_type']}")
        if user_preferences.get("budget_max"):
            pref_items.append(f"Budget limit: ${user_preferences['budget_max']}/hour")
        if pref_items:
            preferences_text = "User preferences:\n" + "\n".join(pref_items)
    
    # Knowledge base info about common GPUs
    gpu_knowledge = """
    GPU KNOWLEDGE BASE:
    - High-end GPUs typically have more VRAM and are better for large ML models
    - GPUs with more vCPUs generally offer better performance for parallel tasks
    - Spot instances can be significantly cheaper but may be interrupted
    - Different regions can have different pricing and availability
    """
    
    # Construct the prompt
    user_prompt = f"""
    USER QUERY: {query}
    
    AVAILABLE GPU OPTIONS:
    {gpu_context}
    
    {preferences_text}
    
    {gpu_knowledge}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",  # Consider "gpt-4" for better results
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        
        response_text = response.choices[0].message.content
        
        try:
            # Parse and validate the response
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {response_text}")
            return {
                "error": "Failed to parse recommendation",
                "raw_response": response_text
            }
            
    except Exception as e:
        logger.error(f"Error generating GPU recommendation: {str(e)}")
        return {"error": f"Failed to generate recommendation: {str(e)}"}

def explain_recommendation(recommendation, details=False):
    """Get a plain text explanation of the recommendation"""
    if not recommendation or "explanation" not in recommendation:
        return "No recommendation available."
        
    if details and "recommendation" in recommendation:
        primary = recommendation["recommendation"].get("primary", {})
        explanation = recommendation.get("explanation", "")
        
        details_text = f"""
        Recommended GPU: {primary.get('name', 'Unknown')}
        
        {explanation}
        
        Rationale: {primary.get('rationale', 'Not provided')}
        """
        
        if "analysis" in recommendation:
            analysis = recommendation["analysis"]
            details_text += f"""
            
            Workload Match: {analysis.get('workload_match', 'Not analyzed')}
            Cost Efficiency: {analysis.get('cost_efficiency', 'Not analyzed')}
            Limitations: {analysis.get('limitations', 'None mentioned')}
            """
            
        return details_text.strip()
    
    return recommendation.get("explanation", "No explanation provided.")