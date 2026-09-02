from transformers import  AutoTokenizer, AutoModelForCausalLM
model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
prompt = "What is the capital of France?"
print(f"Prompt: {prompt}")
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    return_dict=True
)
print(f"Tokenized inputs: {inputs}")
outputs = model.generate(
    **inputs
)
print(f"Generated outputs: {outputs}")
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Answer: {answer}")