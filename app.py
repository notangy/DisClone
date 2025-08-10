import os
from flask import Flask, request, render_template
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

app = Flask(__name__)

checkpoint_path = os.path.abspath("./checkpoint-900")

# Load your fine-tuned model and tokenizer once when the app starts
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token 

model = GPT2LMHeadModel.from_pretrained(checkpoint_path, local_files_only=True)
model.eval()  # set model to evaluation mode

@app.route("/", methods=["GET", "POST"])
def home():
    generated_text = ""
    if request.method == "POST":
        prompt = request.form["prompt"]
        formatted_input = f"Prompt: {prompt}\nResponse:"
        inputs = tokenizer.encode(formatted_input, return_tensors="pt")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = inputs.to(device)
        
        # Generate output (tweak parameters for creativity or length)
        outputs = model.generate(
            inputs, 
            max_new_tokens=50,
            do_sample=True, 
            top_p=0.95, 
            top_k=90,
            temperature=0.8,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        print(outputs)
        generated_sequence = tokenizer.decode(outputs[0])
        print("Full generated sequence:", repr(generated_sequence))
        generated_text = generated_sequence.split("Response:")[-1].strip() 
        print(generated_text)
        
    return render_template("index.html", generated_text=generated_text)

if __name__ == "__main__":
    app.run(port=8000, debug=True)
