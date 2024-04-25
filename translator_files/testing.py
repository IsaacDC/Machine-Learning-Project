import torch
from transformers import MarianMTModel, MarianTokenizer

def translate(input_text, source_lang, target_lang):
    # Load the fine-tuned OPUS-MT model and tokenizer
    model_repo = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_repo)
    model = MarianMTModel.from_pretrained(model_repo)
    model = model.cuda()

    # Tokenize and encode the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    input_ids = input_ids.cuda() if torch.cuda.is_available() else input_ids

    # Translate the input text
    translated_text = model.generate(input_ids)

    # Decode the translated text
    decoded_translation = tokenizer.decode(translated_text[0], skip_special_tokens=True)

    return decoded_translation

# Example usage
input_text = "Hello, how are you?"
translated_text = translate(input_text, "en", "zh")
print("Input text:", input_text)
print("Translated text:", translated_text)

