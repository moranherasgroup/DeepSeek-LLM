import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    print("Loading DeepSeek LLM 7B Base model...")
    
    # Model configuration
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate settings
        print("Loading model (this may take a few minutes)...")
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
        
        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
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
        print("1. Make sure you have sufficient GPU memory (at least 16GB recommended)")
        print("2. Install required packages: pip install torch transformers accelerate")
        print("3. If you get CUDA out of memory, try using CPU: device_map='cpu'")

if __name__ == "__main__":
    main()