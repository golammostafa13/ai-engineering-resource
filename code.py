from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(input_ids)

for id in input_ids[0]:
    print(tokenizer.decode(id))