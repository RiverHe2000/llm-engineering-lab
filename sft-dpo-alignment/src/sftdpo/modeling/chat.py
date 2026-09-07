"""Chat formatting: the single place that decides what the model actually sees.

Two strings have to agree or a fine-tune quietly optimises a distribution the model is
never asked to generate from: the text a training example is built from, and the text the
model is prompted with at evaluation time. Both are produced here, from the same
`ChatFormatter`, so they cannot drift apart.

The formatter delegates to a tokenizer's `apply_chat_template` whenever one is available,
because the special tokens and whitespace of an instruct model's template are part of its
pre-training and guessing them is a known way to lose several points of accuracy. CI has no
Qwen tokenizer offline, so an explicit ChatML fallback is provided and documented: it is the
template Qwen2.5-Instruct publishes, written out in full rather than approximated.

The interesting method is `split_lengths`. Completion-only training needs to know exactly
where the prompt stops and the answer starts *in token space*, which is not simply the
character length of the prompt: a tokenizer is free to merge the last character of the
prompt with the first character of the answer into one token. The split is therefore
computed as the longest common prefix of the two token sequences, which degrades safely --
a merged boundary token is treated as part of the completion and is supervised, rather than
silently shifting every label by one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "IM_END",
    "IM_START",
    "VALID_ROLES",
    "ChatFormatter",
    "ChatTemplateError",
    "common_prefix_length",
    "encode_text",
    "render_chatml",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful financial-advice assistant. Read the adviser's file note and reply "
    "with one JSON object matching the schema. Reply with JSON only, no commentary."
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

VALID_ROLES = ("system", "user", "assistant")


class ChatTemplateError(ValueError):
    """Raised when a tokenizer advertises a chat template that cannot be used.

    Deliberately not a silent fallback to ChatML: if a tokenizer carries a template, that
    template defines the model's expected input, and quietly substituting a different one
    would produce a fine-tune that is subtly mis-formatted everywhere.
    """


def render_chatml(
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    """Render messages in ChatML, the format Qwen2.5-Instruct is trained on.

    Written out explicitly rather than pulled from a tokenizer so that CI -- which has no
    model files -- exercises the same code path shape as a real run.

    Args:
        messages: Ordered turns, each a mapping with `role` and `content` keys.
        add_generation_prompt: Append the open assistant header (`<|im_start|>assistant\\n`)
            so the model's next token is the first token of its answer.

    Returns:
        The rendered conversation.

    Raises:
        ValueError: If there are no messages, or a role is not one of `VALID_ROLES`.
    """
    if not messages:
        raise ValueError("render_chatml needs at least one message")
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        if role not in VALID_ROLES:
            raise ValueError(f"unknown chat role {role!r}; expected one of {VALID_ROLES}")
        parts.append(f"{IM_START}{role}\n{message['content']}{IM_END}\n")
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def encode_text(tokenizer: Any, text: str) -> list[int]:
    """Tokenise `text` without any extra special tokens.

    `add_special_tokens=False` is not a detail: the chat template already contains every
    special token the model expects, and a tokenizer that also prepends BOS would shift the
    prompt/completion boundary by one token and mis-align every label in the batch.
    """
    return list(tokenizer.encode(text, add_special_tokens=False))


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Number of leading elements the two sequences agree on."""
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _template_available(tokenizer: Any) -> bool:
    """Whether `tokenizer` can render chat text itself.

    A tokenizer may expose `apply_chat_template` and still have no template attached (a
    base, non-instruct checkpoint); calling it then raises, so both are checked.
    """
    if tokenizer is None:
        return False
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        return False
    return bool(getattr(tokenizer, "chat_template", None))


@dataclass(frozen=True, slots=True)
class ChatFormatter:
    """Builds the prompt-only prefix and the full training text for one example.

    Attributes:
        system: The system turn. Passed in rather than hard-coded so the task package owns
            the wording and this module owns only the formatting.
        tokenizer: Optional tokenizer used for template rendering when no per-call
            tokenizer is given. Binding one here keeps call sites short in a training loop
            where the tokenizer never changes.
    """

    system: str = DEFAULT_SYSTEM_PROMPT
    tokenizer: Any = None

    def _resolve(self, tokenizer: Any) -> Any:
        return self.tokenizer if tokenizer is None else tokenizer

    def uses_chat_template(self, tokenizer: Any = None) -> bool:
        """Whether rendering will use the tokenizer's template rather than the fallback."""
        return _template_available(self._resolve(tokenizer))

    def messages(self, user_prompt: str, completion: str | None = None) -> list[dict[str, str]]:
        """The conversation as a list of turns, with the answer appended when given."""
        turns = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_prompt},
        ]
        if completion is not None:
            turns.append({"role": "assistant", "content": completion})
        return turns

    def _render(
        self,
        tokenizer: Any,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> str:
        if not _template_available(tokenizer):
            return render_chatml(messages, add_generation_prompt=add_generation_prompt)
        try:
            rendered = tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ChatTemplateError(f"tokenizer chat template failed: {exc}") from exc
        if not isinstance(rendered, str):
            raise ChatTemplateError(
                "tokenizer chat template returned "
                f"{type(rendered).__name__}, expected str; was tokenize=False honoured?"
            )
        return rendered

    def prompt_text(self, user_prompt: str, *, tokenizer: Any = None) -> str:
        """The prompt-only prefix, ending where the model's answer begins.

        This is the string used both for sampling at evaluation time and for locating the
        supervision boundary at training time.
        """
        resolved = self._resolve(tokenizer)
        return self._render(resolved, self.messages(user_prompt), add_generation_prompt=True)

    def full_text(self, user_prompt: str, completion: str, *, tokenizer: Any = None) -> str:
        """The complete training text: prompt prefix plus the answer and its stop token."""
        resolved = self._resolve(tokenizer)
        return self._render(
            resolved,
            self.messages(user_prompt, completion),
            add_generation_prompt=False,
        )

    def encoded_split(
        self,
        tokenizer: Any,
        user_prompt: str,
        completion: str,
    ) -> tuple[list[int], int]:
        """Tokenise the full text once and return it with the prompt token count.

        Returns:
            A pair of (token ids for the full text, number of leading prompt tokens).

        Raises:
            ValueError: If the rendered prompt is not a textual prefix of the rendered full
                text -- a template that reorders or rewrites earlier turns when an assistant
                message is appended makes completion-only masking meaningless, and failing
                loudly beats training on wrong labels.
            ValueError: If the completion contributes no tokens.
        """
        prompt = self.prompt_text(user_prompt, tokenizer=tokenizer)
        full = self.full_text(user_prompt, completion, tokenizer=tokenizer)
        if not full.startswith(prompt):
            raise ValueError(
                "rendered prompt is not a prefix of the rendered full text; "
                "the chat template cannot be used for completion-only masking"
            )
        prompt_ids = encode_text(tokenizer, prompt)
        full_ids = encode_text(tokenizer, full)
        prompt_len = common_prefix_length(prompt_ids, full_ids)
        if prompt_len >= len(full_ids):
            raise ValueError("completion contributes no tokens; refusing to build an example")
        return full_ids, prompt_len

    def split_lengths(
        self,
        tokenizer: Any,
        user_prompt: str,
        completion: str,
    ) -> tuple[int, int]:
        """Token counts of the prompt and of the completion, in that order.

        The two always sum to the length of the tokenised full text, which is the invariant
        the collator relies on when it masks labels.
        """
        full_ids, prompt_len = self.encoded_split(tokenizer, user_prompt, completion)
        return prompt_len, len(full_ids) - prompt_len
