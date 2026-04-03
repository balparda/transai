<!-- cspell:disable -->
<!-- auto-generated; DO NOT EDIT! see base.GenerateTyperHelpMarkdown() -->

# `transai` Command-Line Interface

```text
Usage: transai [OPTIONS] COMMAND [ARGS]...                                                                                                                
                                                                                                                                                           
 AI library and helpers (Python/Poetry/Typer - LM Studio & llama.cpp)                                                                                      
                                                                                                                                                           
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --version                                                        Show version and exit.                                                                 │
│ --verbose             -v                INTEGER RANGE [0<=x<=3]  Verbosity (nothing=ERROR, -v=WARNING, -vv=INFO, -vvv=DEBUG).               │
│ --color                   --no-color                             Force enable/disable colored output (respects NO_COLOR env var if not provided).       │
│                                                                  Defaults to having colors.                                                             │
│ --foo                 -f                INTEGER                  Some integer option.                                                    │
│ --bar                 -b                TEXT                     Some string option.                                              │
│ --install-completion                                             Install completion for the current shell.                                              │
│ --show-completion                                                Show completion for the current shell, to copy it or customize the installation.       │
│ --help                                                           Show this message and exit.                                                            │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ markdown  Emit Markdown docs for the CLI (see README.md section "Creating a New Version").                                                              │
│ random    Random utilities.                                                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
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

## `transai random` Command

```text
Usage: transai random [OPTIONS] COMMAND [ARGS]...                                                                                                         
                                                                                                                                                           
 Random utilities.                                                                                                                                         
                                                                                                                                                           
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ num  Generate a random integer.                                                                                                                         │
│ str  Generate a random string.                                                                                                                          │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### `transai random num` Sub-Command

```text
Usage: transai random num [OPTIONS]                                                                                                                       
                                                                                                                                                           
 Generate a random integer.                                                                                                                                
                                                                                                                                                           
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --min         INTEGER  Minimum value (inclusive).                                                                                           │
│ --max         INTEGER  Maximum value (inclusive).                                                                                         │
│ --help                 Show this message and exit.                                                                                                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### `transai random str` Sub-Command

```text
Usage: transai random str [OPTIONS]                                                                                                                       
                                                                                                                                                           
 Generate a random string.                                                                                                                                 
                                                                                                                                                           
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --length    -n      INTEGER RANGE   String length.                                                                                   │
│ --alphabet          TEXT                  Custom alphabet to sample from (defaults to ).                                                                │
│ --help                                    Show this message and exit.                                                                                   │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
