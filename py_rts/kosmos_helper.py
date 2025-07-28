
# Clean up the output
def clean_response(text):
    # Remove special tokens and formatting
    text = text.replace('<grounding>', '').replace('</grounding>', '')
    text = text.replace('<object>', '').replace('</object>', '')
    # Capitalize first letter and ensure proper punctuation
    text = text.strip()
    text = text[0].upper() + text[1:]
    if not text.endswith('.'):
        text += '.'
    return text

final_response = clean_response(generated_text)
print("Cleaned Response:", final_response)