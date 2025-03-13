import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Constants
MODEL_NAME = "deepseek-ai/deepseek-math-7b-base"
SAVE_PATH = "finetuned_deepseek_math"

@st.cache_resource(show_spinner=False)
def load_model():
    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load tokenizer and model in 4-bit mode
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config
    )
    model.generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Prepare model for k-bit training and wrap with LoRA via PEFT
    model = prepare_model_for_kbit_training(model)
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