import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import LoraConfig, get_peft_model

# Constants
MODEL_NAME = "deepseek-ai/deepseek-math-7b-base"
SAVE_PATH = "finetuned_deepseek_math"

@st.cache_resource(show_spinner=False)
def load_model():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the full-precision model without quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Using full precision on CPU
        device_map="cpu"
    )
    model.generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Attach LoRA adapters
    lora_config = LoraConfig(
        r=20,
        lora_alpha=40,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Load the fine-tuned adapter weights
    model.load_pretrained(SAVE_PATH)
    model.eval()
    return tokenizer, model

def generate_output(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=20,
            generation_config=model.generation_config
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

st.title("Deepseek Math Fine-Tuned Model Inference")
st.write("Enter your prompt below:")

user_input = st.text_input("Prompt", "🚗 + 🚗 + 🚗 + 🚗 = 20 → 🚗 =")

if st.button("Generate Output"):
    with st.spinner("Generating answer..."):
        tokenizer, model = load_model()
        output = generate_output(user_input, tokenizer, model)
    st.success("Output generated!")
    st.write("**Input:**", user_input)
    st.write("**Output:**", output)