# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Load llm utilities."""

from __future__ import annotations

import logging
from graphrag.config.enums import LLMType
from graphrag.config.models.language_model_config import (
    LanguageModelConfig,  # noqa: TC001
)
from graphrag.index.llm.manager import ChatLLMSingleton, EmbeddingsLLMSingleton
from graphrag.index.llm.openai_interface import AsyncOpenAIClient

log = logging.getLogger(__name__)

def load_llm(
    name: str,
    config: LanguageModelConfig,
    *,
    chat_only=False,
) -> AsyncOpenAIClient:
    """Load the LLM for the entity extraction chain."""
    singleton_llm = ChatLLMSingleton().get_llm(name)
    if singleton_llm is not None:
        return singleton_llm

    llm_type = config.type

    if llm_type in loaders:
        if chat_only and not loaders[llm_type]["chat"]:
            msg = f"LLM type {llm_type} does not support chat"
            raise ValueError(msg)

        loader = loaders[llm_type]
        llm_instance = loader["load"](config)
        ChatLLMSingleton().set_llm(name, llm_instance)
        return llm_instance

    msg = f"Unknown LLM type {llm_type}"
    raise ValueError(msg)


def load_llm_embeddings(
    name: str,
    llm_config: LanguageModelConfig,
    *,
    chat_only=False,
):
    """Load the LLM for the entity extraction chain."""
    singleton_llm = EmbeddingsLLMSingleton().get_llm(name)
    if singleton_llm is not None:
        return singleton_llm

    llm_type = llm_config.type
    if llm_type in loaders:
        if chat_only and not loaders[llm_type]["chat"]:
            msg = f"LLM type {llm_type} does not support chat"
            raise ValueError(msg)
        llm_instance = loaders[llm_type]["load"](llm_config)
        EmbeddingsLLMSingleton().set_llm(name, llm_instance)
        return llm_instance

    msg = f"Unknown LLM type {llm_type}"
    raise ValueError(msg)

def _load_openai_chat_llm(config: LanguageModelConfig):
    return AsyncOpenAIClient(config)

def _load_openai_embeddings_llm(config: LanguageModelConfig):
    pass

loaders = {
    LLMType.OpenAIChat: {
        "load": _load_openai_chat_llm,
        "chat": True,
    },
    LLMType.OpenAIEmbedding: {
        "load": _load_openai_embeddings_llm,
        "chat": False,
    },
}

