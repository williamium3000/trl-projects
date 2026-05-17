"""Model-family-specific processor loading for mllm-co-grpo-dp.

Some VLM checkpoints ship a `processor_config.json` / `chat_template.jinja`
that is out of sync with the `AutoProcessor` class shipped in
transformers 4.57.x. Loading them through `AutoProcessor.from_pretrained`
directly raises before the processor is even constructed.

Concretely, **InternVL3.5-4B** (and InternVL2.5-* / InternVL3-*) trip on:

1. `transformers.models.internvl.InternVLProcessor.__init__` reads four named
   attributes off the tokenizer (`start_image_token`, `end_image_token`,
   `context_image_token`, `video_token`, plus their `*_id` siblings). The
   InternVL3.5-4B tokenizer (a `Qwen2TokenizerFast`) adds the corresponding
   special tokens via `additional_special_tokens` but never binds them to
   named attributes → `AttributeError` at construction time.

2. The shipped `chat_template.jinja` renders an `{"type": "image"}` block
   as the literal string `<image>\n`, but the processor's
   `_insert_media_placeholders` searches for `<IMG_CONTEXT>` (the model's
   image_token). With the template unpatched, `processor(images=..., text=...)`
   raises `ValueError: Number of image placeholders ... does not match`.

Both are local data-bag mismatches (no model-architecture change required),
fixed here by:

(1) Loading the tokenizer first, monkey-patching the four `*_token` /
    `*_token_id` attributes from `added_tokens.json`, then handing the
    patched tokenizer to `AutoProcessor.from_pretrained(..., tokenizer=tok)`.

(2) Replacing the literal `'<image>\n'` substring in `chat_template` with
    `'<IMG_CONTEXT>\n'` so `apply_chat_template` emits a placeholder the
    processor's `_insert_media_placeholders` actually recognizes.

Qwen2.5-VL (`Qwen/Qwen2.5-VL-*-Instruct`) does not need any of this — it
ships a Qwen2VLProcessor with consistent tokenizer bindings and a
chat_template that already emits `<|vision_start|><|image_pad|><|vision_end|>`.
For Qwen2.5-VL we fall through to the plain `AutoProcessor.from_pretrained`
path.

Verified 2026-05-17 (marti-parity env: transformers 4.57.6, vllm 0.18.0):
  end-to-end `apply_chat_template` → `processor(images=PIL, text=...)`
  yields `input_ids.shape=(1, 270)`, `pixel_values.shape=(1, 3, 448, 448)`,
  with 256 `<IMG_CONTEXT>` (id 151671) tokens — matching InternVL's
  expected image_seq_length of 256.
"""

from transformers import AutoProcessor, AutoTokenizer


_INTERNVL_TOKEN_ATTRS = {
    # AttrName              → token literal in InternVL3.x special_tokens_map
    "start_image_token":     "<img>",
    "end_image_token":       "</img>",
    "context_image_token":   "<IMG_CONTEXT>",
    "video_token":           "<|video_pad|>",
}


def _is_internvl(model_name_or_path: str) -> bool:
    """Heuristic on HF repo name. Cheap; avoids an extra AutoConfig round-trip."""
    name = model_name_or_path.lower()
    return ("internvl" in name) and ("opengvlab" in name or "intern" in name)


def _patch_internvl_tokenizer(tokenizer):
    """Bind the four image/video special-token attributes the InternVLProcessor reads.

    Raises if any expected token is not present in the tokenizer's vocab — which
    would indicate a non-InternVL-3.x tokenizer was passed.
    """
    for attr, literal in _INTERNVL_TOKEN_ATTRS.items():
        tid = tokenizer.convert_tokens_to_ids(literal)
        if tid is None or tid == tokenizer.unk_token_id:
            raise RuntimeError(
                f"InternVL tokenizer is missing special token {literal!r}; "
                f"got id={tid}. Cannot patch processor."
            )
        setattr(tokenizer, attr, literal)
        setattr(tokenizer, f"{attr}_id", tid)


def _patch_internvl_chat_template(processor):
    """Rewrite the chat_template to emit `<IMG_CONTEXT>` placeholders.

    The processor's `_insert_media_placeholders` only recognizes
    `processor.image_token` (= `<IMG_CONTEXT>`) when scanning prompts; the
    shipped template emits `<image>` which the processor cannot match.
    Patch both `processor.tokenizer.chat_template` and `processor.chat_template`
    because TRL may read from either.
    """
    needle = "'<image>\n'"
    replacement = "'<IMG_CONTEXT>\n'"
    for owner in (processor.tokenizer, processor):
        tpl = getattr(owner, "chat_template", None)
        if not tpl or needle not in tpl:
            continue
        owner.chat_template = tpl.replace(needle, replacement)


def load_processor_for_mllm(model_name_or_path, **kwargs):
    """Return an `AutoProcessor` for `model_name_or_path`, applying any
    family-specific patches required to run inside marti-parity env
    (transformers 4.57.6 + vllm 0.18 + no math_verify).

    Args:
        model_name_or_path: HF repo or local path.
        **kwargs: forwarded to `AutoProcessor.from_pretrained` for the
            non-patched path, and split between `AutoTokenizer.from_pretrained`
            and `AutoProcessor.from_pretrained` for the InternVL path
            (`padding_side`, `trust_remote_code`, `revision` are honored on
            both calls).

    Returns:
        A ready-to-use processor whose `__call__(images=..., text=...)` will
        not raise for chat-template-rendered prompts.
    """
    if not _is_internvl(model_name_or_path):
        return AutoProcessor.from_pretrained(model_name_or_path, **kwargs)

    # Pull out kwargs that AutoTokenizer also accepts.
    tok_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ("revision", "trust_remote_code", "padding_side")
    }
    # AutoTokenizer does not accept padding_side via kwarg in all versions;
    # set it on the instance instead.
    padding_side = tok_kwargs.pop("padding_side", None)
    # InternVL requires trust_remote_code=True for its custom modeling stub,
    # but the config-only path (which AutoTokenizer uses) does not strictly
    # need it. Pass through if caller set it.
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
    if padding_side is not None:
        tokenizer.padding_side = padding_side

    _patch_internvl_tokenizer(tokenizer)

    proc_kwargs = dict(kwargs)
    proc_kwargs["tokenizer"] = tokenizer
    # `trust_remote_code` defaults False; many InternVL checkpoints still
    # need it for the chat-template / config side.
    proc_kwargs.setdefault("trust_remote_code", True)
    processor = AutoProcessor.from_pretrained(model_name_or_path, **proc_kwargs)

    _patch_internvl_chat_template(processor)

    return processor


__all__ = ["load_processor_for_mllm"]
