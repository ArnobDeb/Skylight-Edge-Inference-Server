# OpenVINO GenAI – Model Loading

## What LLMPipeline() does

Creating an LLMPipeline does **not** generate any text.

It performs initialization:

1. Reads model configuration.
2. Loads tokenizer.
3. Loads detokenizer.
4. Reads OpenVINO IR (.xml + .bin).
5. Creates OpenVINO Runtime Core.
6. Compiles the model for the selected device.
7. Allocates runtime resources.

After this step, the pipeline is ready to perform inference.

Inference starts only when generate() is called.

When Python executed

pipeline = ov_genai.LLMPipeline(
    str(MODEL_PATH),
    device="CPU"
)

OpenVINO GenAI roughly performed the following sequence:

                 LLMPipeline(...)
                        │
                        ▼
          Read configuration files
                        │
                        ▼
        Load tokenizer & detokenizer
                        │
                        ▼
          Create OpenVINO Runtime Core
                        │
                        ▼
       Read openvino_model.xml/.bin
                        │
                        ▼
         Compile model for CPU plugin
                        │
                        ▼
         Allocate runtime resources
                        │
                        ▼
      Pipeline ready for text generation

Notice that no neural network inference has happened yet. We've only prepared the runtime.
