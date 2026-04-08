<!-- cspell:disable -->
<!-- auto-generated; DO NOT EDIT! see base.GenerateTyperHelpMarkdown() -->

# `transai` Command-Line Interface

```text
Usage: transai [OPTIONS] COMMAND [ARGS]...                                                                                                                
                                                                                                                                                           
 AI library and helpers (Python/Poetry/Typer - LM Studio & llama.cpp)                                                                                      
                                                                                                                                                           
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --version                                                                 Show version and exit.                                                        │
│ --verbose             -v                INTEGER RANGE [0<=x<=3]           Verbosity (nothing=ERROR, -v=WARNING, -vv=INFO, -vvv=DEBUG).      │
│ --color                   --no-color                                      Force enable/disable colored output (respects NO_COLOR env var if not         │
│                                                                           provided). Defaults to having colors.                                         │
│ --root                -r                DIRECTORY                         The local machine models root directory path, ex: "~/.lmstudio/models/"; will │
│                                                                           expect the models to be in directories under this given root, usually the     │
│                                                                           hierarchy looks like: <root>/<user>/<model-dir>/*.gguf; only necessary for    │
│                                                                           non-LMStudio (`--no-lms`) runs; default: the LMStudio models root if it       │
│                                                                           exists, otherwise no default and you must provide it                          │
│                                                                                                              │
│ --lms                     --no-lms                                        Use LMStudio client library for AI instead of the old llama-cpp-python        │
│                                                                           library? default: True (LMStudio)                                             │
│                                                                                                                                           │
│ --model               -m                TEXT                              LLM model to load and use: the model must be compatible with the              │
│                                                                           llama.cpp/LMStudio client libraries; will NOT get the model for you, so make  │
│                                                                           sure you either have it available in your LMStudio or the model files are     │
│                                                                           under the specified models root path (`-r/--root` option); should be a string │
│                                                                           you would use with `lms get <THIS>` or `https://huggingface.co/<THIS>`;       │
│                                                                           default: 'qwen3-8b@Q8_0', a good general-purpose text (non-vision) model      │
│                                                                                                                                 │
│ --tokens              -t                INTEGER RANGE [2<=x<=200]         Speculative Decoding: controls how many tokens the model should generate in   │
│                                                                           advance during auto-tagging; if you do not define this flag then speculative  │
│                                                                           decoding will be disabled; usually this is a small value, like 4 or 8, and it │
│                                                                           can improve the speed of processing by allowing the model to generate tokens  │
│                                                                           in parallel; default: None (disabled)                                         │
│ --seed                -s                INTEGER RANGE [2<=x<=2147483647]  A seed value for the random number generator used to load the models into     │
│                                                                           memory; providing a seed ensures reproducibility of the results; default:     │
│                                                                           None (randomized seed)                                                        │
│ --context                               INTEGER RANGE [16<=x<=16777216]   Maximum number of tokens to use as context for the model; default: 32768      │
│                                                                           tokens                                                                        │
│                                                                                                                                         │
│ --temperature         -x                FLOAT RANGE [0.0<=x<=2.0]         Temperature controls how random token selection is during generation; [0 or   │
│                                                                           near 0]: most deterministic, focused, repetitive, best for extraction /       │
│                                                                           structured output / coding / tool use; [0.2-0.5]: still stable, but less      │
│                                                                           rigid; [0.7-1.0]: more natural and varied; [>1.0]: often more creative, but   │
│                                                                           also more errors, drift, and nonsense; default: 0.150 (a good value for       │
│                                                                           structured output and tool use)                                               │
│                                                                                                                                          │
│ --gpu                 -g                FLOAT RANGE [0.1<=x<=1.0]         GPU ratio to use, a value between 0.1 (10%) and 1.0 (100%) that indicates the │
│                                                                           percentage of GPU resources to allocate to AI; default: 0.80                  │
│                                                                                                                                           │
│ --gpu-layers                            INTEGER RANGE [-1<=x<=128]        Number of layers offloaded to GPU; default: -1 (which means "as many as       │
│                                                                           possible")                                                                    │
│                                                                                                                                            │
│ --fp16                    --no-fp16                                       Use FP16 precision for the auto-tagger model? This can reduce memory usage    │
│                                                                           and potentially increase speed, but may slightly affect the accuracy of the   │
│                                                                           tagging results default: False (do not use FP16, use full precision)          │
│                                                                                                                                       │
│ --mmap                    --no-mmap                                       Use memory-mapped file loading (if supported)? default: True (use mmap)       │
│                                                                                                                                          │
│ --flash                   --no-flash                                      Enable flash attention (if supported)? default: True (use flash)              │
│                                                                                                                                         │
│ --kv-cache                              INTEGER RANGE [4<=x<=128]         GGML type for KV-cache keys/values (if supported): determines the precision   │
│                                                                           level used to store keys/values; default: None (store according to original   │
│                                                                           precision in model)                                                           │
│ --timeout                               FLOAT RANGE [1.0<=x<=86400.0]     Timeout in seconds for model loading and calls; default: 5.000 min            │
│                                                                                                                                         │
│ --install-completion                                                      Install completion for the current shell.                                     │
│ --show-completion                                                         Show completion for the current shell, to copy it or customize the            │
│                                                                           installation.                                                                 │
│ --help                                                                    Show this message and exit.                                                   │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ markdown  Emit Markdown docs for the CLI (see README.md section "Creating a New Version").                                                              │
│ query     Query the model.                                                                                                                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                                                                                                                           
 Examples:                                                                                                                                                 
                                                                                                                                                           
 # --- Query the AI ---                                                                                                                                    
 poetry run transai query "What is the capital of France?"                                                                                                 
 poetry run transai --no-lms query "Give me an onion soup recipe."                                                                                         
                                                                                                                                                           
 # --- Markdown ---                                                                                                                                        
 poetry run transai markdown > transai.md
```

## `transai markdown` Command

```text
Usage: transai markdown [OPTIONS]                                                                                                                         
                                                                                                                                                           
 Emit Markdown docs for the CLI (see README.md section "Creating a New Version").                                                                          
                                                                                                                                                           
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                                                                                                                           
 Example:                                                                                                                                                  
                                                                                                                                                           
 $ poetry run transai markdown > transai.md                                                                                                                
 <<saves CLI doc>>
```

## `transai query` Command

```text
Usage: transai query [OPTIONS] MODEL_INPUT                                                                                                                
                                                                                                                                                           
 Query the model.                                                                                                                                          
                                                                                                                                                           
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_input      TEXT  Query input string; "user prompt"                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --system  -y                TEXT  Prefix prompt; prepend to query; "system prompt"; default: no system prompt                                           │
│ --images  -i                PATH  A list of image paths to use as input for the model query; default: None, no images                                   │
│ --tools   -z                TEXT  A list of python methods to use as tools for the model query; default: None, no tools                                 │
│ --free        --no-free           Unload previous models before loading new ones (LM Studio)? default: False (keep)                   │
│ --metal       --no-metal          Print Metal/llama.cpp verbose internals? default: False (do not print)                             │
│ --help                            Show this message and exit.                                                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                                                                                                                           
 Example:                                                                                                                                                  
                                                                                                                                                           
 poetry run transai query "What is the capital of France?"                                                                                                 
 poetry run transai --no-lms query "Give me an onion soup recipe."
```
