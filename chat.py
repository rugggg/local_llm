from mlx_lm import load, generate, stream_generate
import gradio as gr
model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")


def gr_chat_wrapper(message, history):
    local_hist = ""
    prompt_lead = "<s> [INST] You are PotatoBot, a not very smart, but very emotionally savvy and helpful organic robot. You love potatos in all their forms and will mention them as much as possible. Answer the question very succinctly. [INST]</s>" 
    for t in stream_generate(model, tokenizer, prompt=f"{prompt_lead} {message}", max_tokens=512):
        local_hist = f'{local_hist}{t}'
        yield local_hist

gr.ChatInterface(gr_chat_wrapper).launch(server_name='0.0.0.0')
