from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time

def warmup(model, tokenizer):
    warmup_prompt = "This is me warming up the model"
    encoded = tokenizer.encode_plus(
        warmup_prompt,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    model_inputs = []
    attention_inputs = []
    
    model_inputs.append(encoded['input_ids']) # Collect the model_inputs
    attention_inputs.append(encoded['attention_mask'])  # Collect the attention mask
    
    model_inputs = torch.cat(model_inputs).to("cuda") #Concatenate the model_inputs
    attention_inputs = torch.cat(attention_inputs).to("cuda")  # Concatenate attention masks

    generated_ids = model.generate(input_ids=model_inputs, attention_mask=attention_inputs, max_new_tokens=128, do_sample=True) # Setting max_new_tokens to 128 as per PS
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])
    
def main(model_id='mistralai/Mistral-7B-Instruct-v0.1'):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto", quantization_config = quant_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    warmup(model, tokenizer)
    
    prompt = input("Enter your prompt: ")
    encoded = tokenizer.encode_plus(
        prompt,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    model_inputs = []
    attention_inputs = []
    
    model_inputs.append(encoded['input_ids']) # Collect the model_inputs
    attention_inputs.append(encoded['attention_mask'])  # Collect the attention mask
    
    model_inputs = torch.cat(model_inputs).to("cuda") #Concatenate the model_inputs
    attention_inputs = torch.cat(attention_inputs).to("cuda")  # Concatenate attention masks
        
    # Generate output
    start = time.time()
    generated_ids = model.generate(input_ids=model_inputs, attention_mask=attention_inputs, max_new_tokens=128, do_sample=True) # Setting max_new_tokens to 128 as per PS
    end = time.time()
    
    # Calculate latency and throughput
    output_token_count = generated_ids.size(dim=1)
    
    latency = (end - start) # (end - start) gives us the total time taken for generation
    through_put = (output_token_count) / (end - start) # output_token_count is sum of both input tokens (128) and output tokens (128)

    print(f"Latency: {latency} seconds")
    print(f"Throughput: {through_put} tokens/second")

    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])

if __name__ == '__main__':
    torch.set_default_device("cuda")
    model_id = input("Provide model id: ")
    main(model_id)
    torch.cuda.empty_cache()