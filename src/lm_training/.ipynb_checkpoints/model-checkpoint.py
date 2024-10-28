from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForCausalLM
from typing import Union


def initialize_model(
    vocab_size: int, model_type: str, is_mlm: bool = True, **config
) -> Union[AutoModelForMaskedLM, AutoModelForCausalLM]:
    config = AutoConfig.from_pretrained(
        model_type,
        vocab_size=vocab_size,
        **config,
    )

    auto_model = AutoModelForMaskedLM if is_mlm else AutoModelForCausalLM

    model = auto_model.from_config(config)

    return model
