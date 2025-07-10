import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    print("Loading DeepSeek LLM 67B Base model...")
    
    # Model configuration - Updated to 67B model
    model_name = "deepseek-ai/deepseek-llm-67b-base"
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate settings
        print("Loading model (this may take several minutes due to model size)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure generation settings
        model.generation_config = GenerationConfig.from_pretrained(model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        
        # Input text about attention mechanism
        text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
        
        print(f"\nInput text: {text}")
        print("\nGenerating completion...")
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt")
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs.to(model.device), 
                max_new_tokens=100
            )
        
        # Decode the result
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nComplete result:\n{result}")
        
        # Extract only the generated part
        generated_text = result[len(text):]
        print(f"\nGenerated continuation:\n{generated_text}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. The 67B model requires significant GPU memory (80GB+ recommended)")
        print("2. Consider using model sharding across multiple GPUs")
        print("3. If you get CUDA out of memory, try the 7B model instead")
        print("4. Ensure you have sufficient disk space for the model download (~130GB)")

if __name__ == "__main__":
    main()