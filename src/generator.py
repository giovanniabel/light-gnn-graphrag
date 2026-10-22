"""Reader via Ollama (agent.md Sections 4.4, 10).

Every method gets the SAME prompt template and the same K=4 paragraph budget, so
differences in EM/F1 come from routing only (Section 4.4).
"""

from __future__ import annotations

import time

from .ollama_client import OllamaClient
from .schemas import Paragraph, Usage

MAX_TOKENS = 24

SYSTEM = ("You answer questions from the given paragraphs. "
          "Reply with the shortest possible answer span, no explanation.")

# One fixed template for every method. Changing it after seeing results would be a
# deviation (Section 12).
TEMPLATE = """{triple_header}Paragraphs:
{paragraphs}

Question: {question}
Answer with the shortest span only.
Answer:"""


def build_prompt(question: str, paragraphs: list[Paragraph], triples=None,
                 max_words: int = 200) -> str:
    paras = "\n\n".join(
        f"[{i+1}] {p.title}. {' '.join(p.text.split()[:max_words])}"
        for i, p in enumerate(paragraphs)
    )
    header = ""
    if triples:
        lines = "; ".join(f"{t.head} {t.rel} {t.tail}" for t in triples[:5])
        header = f"Facts: {lines}\n\n"
    return TEMPLATE.format(triple_header=header, paragraphs=paras, question=question)


def answer(question: str, paragraphs: list[Paragraph], triples=None,
           model_tag: str | None = None, client=None,
           max_tokens: int = MAX_TOKENS) -> tuple[str, Usage]:
    """Section 10 signature. Greedy, short, one call."""
    import os

    cli = client or OllamaClient(model_tag or os.environ["SLM_TAG"],
                                 num_ctx=2048, temperature=0.0)
    prompt = build_prompt(question, paragraphs, triples)
    text, usage, _raw = cli.generate(prompt, system=SYSTEM, num_predict=max_tokens,
                                     stop=["\n"])
    return text.strip(), usage
