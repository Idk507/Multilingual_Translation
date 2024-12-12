import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TranslationModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
   
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
    
    def translate(self, text, max_length=128):
       
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs, 
            max_length=max_length, 
            num_beams=4, 
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def batch_translate(self, texts, max_length=128):
        
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(self.device)
        
   
        outputs = self.model.generate(
            **inputs, 
            max_length=max_length, 
            num_beams=4, 
            early_stopping=True
        )
        
    
        return [
            self.tokenizer.decode(output, skip_special_tokens=True) 
            for output in outputs
        ]


MODEL_PATH = "./tamil_english_translation_model"

translator = TranslationModel(MODEL_PATH)


tamil_text = "வணக்கம் உலகம்!"
english_translation = translator.translate(tamil_text)
print(f"Tamil: {tamil_text}")
print(f"English: {english_translation}")

# Batch translation
tamil_texts = [
    "வணக்கம் உலகம்!",
    "நான் ஒரு மொழிபெயர்ப்பு மாதிரி உருவாக்குகிறேன்",
    "தமிழ் மொழி அருமையான மொழி"
]
batch_translations = translator.batch_translate(tamil_texts)

print("\nBatch Translation:")
for tamil, english in zip(tamil_texts, batch_translations):
    print(f"Tamil: {tamil}")
    print(f"English: {english}")
    print("---")

