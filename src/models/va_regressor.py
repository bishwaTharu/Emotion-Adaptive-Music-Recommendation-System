from transformers import AutoModel
from .lora import apply_lora_to_transformer
from .wide_deep import WideDeepVA

def build_model(transformer_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> WideDeepVA:
    transformer = AutoModel.from_pretrained(transformer_name)

    for p in transformer.parameters():
        p.requires_grad = False

    apply_lora_to_transformer(transformer)

    model = WideDeepVA(transformer)
    return model
