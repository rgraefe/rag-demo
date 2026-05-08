from typing import Any, Callable, Optional

from llama_index.core import Settings
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_REFINE_PROMPT_SEL,
    DEFAULT_TEXT_QA_PROMPT_SEL,
    DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)
from llama_index.core.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from llama_index.core.response_synthesizers import (
    Accumulate,
    BaseSynthesizer,
    CompactAndRefine,
    Generation,
    Refine,
    SimpleSummarize,
    TreeSummarize,
)
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.types import BasePydanticProgram

from src.synthesizers.type import CiteResponseMode
from src.synthesizers.accumulate_cite import AccumulateCite
from src.synthesizers.compact_cite import CompactCite
from .cite_prompt_selectors import (
    CITATION_TEXT_QA_PROMPT_SEL,
    CITATION_REFINE_PROMPT_SEL,
)


def get_response_synthesizer(
    llm: Optional[LLM] = None,
    prompt_helper: Optional[PromptHelper] = None,
    text_qa_template: Optional[BasePromptTemplate] = None,
    refine_template: Optional[BasePromptTemplate] = None,
    summary_template: Optional[BasePromptTemplate] = None,
    simple_template: Optional[BasePromptTemplate] = None,
    response_mode: CiteResponseMode | ResponseMode = CiteResponseMode.CITE_ACCUMULATE,
    callback_manager: Optional[CallbackManager] = None,
    use_async: bool = False,
    streaming: bool = False,
    structured_answer_filtering: bool = False,
    output_cls: Optional[type[BaseModel]] = None,
    program_factory: Optional[Callable[[BasePromptTemplate], BasePydanticProgram]] = None,
    verbose: bool = False,
) -> BaseSynthesizer:
    """Get a response synthesizer."""

    llm = llm or Settings.llm
    callback_manager = callback_manager or Settings.callback_manager

    prompt_helper = (
        prompt_helper
        or Settings._prompt_helper
        or PromptHelper.from_llm_metadata(llm.metadata)
    )

    def_text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL
    def_refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
    def_simple_template = simple_template or DEFAULT_SIMPLE_INPUT_PROMPT
    def_summary_template = summary_template or DEFAULT_TREE_SUMMARIZE_PROMPT_SEL

    cite_text_qa_template = text_qa_template or CITATION_TEXT_QA_PROMPT_SEL
    cite_refine_template = refine_template or CITATION_REFINE_PROMPT_SEL

    if response_mode == CiteResponseMode.CITE_ACCUMULATE:
        return AccumulateCite(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            text_qa_template=cite_text_qa_template,
            output_cls=output_cls,
            streaming=streaming,
            use_async=use_async,
        )

    if response_mode == CiteResponseMode.CITE_COMPACT:
        return CompactCite(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            text_qa_template=cite_text_qa_template,
            refine_template=cite_refine_template,
            output_cls=output_cls,
            streaming=streaming,
            verbose=verbose,
            structured_answer_filtering=structured_answer_filtering,
            program_factory=program_factory,
        )

    if response_mode == ResponseMode.REFINE:
        return Refine(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            text_qa_template=def_text_qa_template,
            refine_template=def_refine_template,
            output_cls=output_cls,
            streaming=streaming,
            structured_answer_filtering=structured_answer_filtering,
            program_factory=program_factory,
            verbose=verbose,
        )

    if response_mode == ResponseMode.COMPACT:
        return CompactAndRefine(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            text_qa_template=def_text_qa_template,
            refine_template=def_refine_template,
            output_cls=output_cls,
            streaming=streaming,
            structured_answer_filtering=structured_answer_filtering,
            program_factory=program_factory,
            verbose=verbose,
        )

    if response_mode == ResponseMode.TREE_SUMMARIZE:
        return TreeSummarize(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            summary_template=def_summary_template,
            output_cls=output_cls,
            streaming=streaming,
            use_async=use_async,
            verbose=verbose,
        )

    if response_mode == ResponseMode.SIMPLE_SUMMARIZE:
        return SimpleSummarize(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            text_qa_template=def_text_qa_template,
            streaming=streaming,
        )

    if response_mode == ResponseMode.GENERATION:
        return Generation(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            simple_template=def_simple_template,
            streaming=streaming,
        )

    if response_mode == ResponseMode.ACCUMULATE:
        return Accumulate(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            text_qa_template=def_text_qa_template,
            output_cls=output_cls,
            streaming=streaming,
            use_async=use_async,
        )

    raise ValueError(f"Unknown mode: {response_mode}")