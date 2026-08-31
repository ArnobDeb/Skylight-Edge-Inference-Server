# Streaming Generation

The streamer callback is invoked repeatedly during generation.

Observations:

- The callback is not invoked once per word.
- The callback is not invoked once per character.
- It receives incrementally detokenized text fragments.

For the prompt:

"What is OpenVINO?"

Input tokens:
7

Output callback invocations:
49

Measured on CPU:

TTFT ≈ 6 seconds

Inter-callback latency ≈ 116 ms

Generation latency ≈ 11.6 seconds

This suggests:

1. Prompt processing dominates TTFT.
2. After the first generated token, decoding proceeds at a relatively constant rate.