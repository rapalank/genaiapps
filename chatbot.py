from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)


    input_text = input("> ") 

    history_string = '\n'.join(conversation_history)
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors = "pt")

    #print(inputs)


    outputs = model.generate(**inputs)
    #print(outputs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response)

    conversation_history.append(input_text)
    conversation_history.append(response)