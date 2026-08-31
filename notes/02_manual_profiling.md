# Manual Profiling

The first profiler separates model initialization from text generation.

## Pipeline Initialization

Includes:

- Reading model files
- Loading tokenizer
- Loading detokenizer
- Creating OpenVINO Runtime Core
- Compiling model for target device
- Runtime initialization

This happens only once.

## Generation Time

Includes:

- Prompt tokenization
- Transformer inference
- Autoregressive decoding
- Detokenization

Generation time depends on:

- Prompt length
- Number of generated tokens
- Model size
- Hardware

Initialization time and generation time should always be measured separately.
