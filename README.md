# LocalGPUpdfSummarizer

PDF summarizer suitable to run fully free and locally on GPUs of various ages/types, including older NVIDIA GPUs. The main implementation exemplified within this scripting utilizes the RPC back end of llama.cpp to "pool" vRAM across a GTX 970 GPU plus a GTX 1050 notebook GPU, from separate computers connected via an Ethernet cable.

Designed with the principle of scientific trustworthiness in mind, Perplexity AI was consulted while programming this codebase (a preferable choice in terms of scientific trustworthiness, thanks to citation comprehension). The primary aim of this repo is to go beyond the standard "run-of-the-mill" or "hype" stemming from most LLM based PDF summarization attempts. Scientific trustworthiness may not be possible to guarantee, but this repo strives towards it.

While the particular hardware considerations encoded into both of the primary files within this repo are most pertinent to the local GPU configuration available for the main uploader, the presets/parameters specific to the GTX 970 GPU and GTX 1050 notebook GPU are relatively minimal (and not too difficult to substitute for better personalization).

All dependencies, including a fresh compilation of `llama.cpp`, are wrapped into Pixi via the config file `pixi.toml`. The remainder of the codebase is located within the Python script `summarizer.py`.

No GPU is technically required, thanks to CPU offloading handled by `llama.cpp`. This script has been tested and successfully can run in CPU only mode (prioritizing scientific trustworthiness over speed, expecting longer run times without GPU acceleration).

If Pixi is not already installed, see documentation at: https://pixi.prefix.dev/latest/installation/

To run locally, the `[tasks]` section of `pixi.toml` allows for several `llama.cpp` building commands. Assuming a 2 computer configuration with an Ethernet connection specifically featuring a GTX 970 GPU and a GTX 1050 notebook GPU, a singular commit/version of `llama.cpp` must be stated (under `LLAMA_COMMIT` in `pixi.toml`). The already chosen commit can be tested, or a newer version is likely to be supported as well. The following command sequence is recommended (at least for similar or identical hardware as stated above):

```
git clone https://github.com/TromboneEngineer/LocalGPUpdfSummarizer.git
cd LocalGPUpdfSummarizer/
pixi run clone
pixi run pin
pixi run build
```

Then once this repo and a fresh build of `llama.cpp` are available (note: this may need to be carried out on each machine utilized, if the RPC back end of `llama.cpp` is opted for), some recommended commands to run the local GPU PDF summarizer:

First, within a Tmux session on the RPC host machine (the GTX 970 GPU containing machine is selected within local testing):

```
cd LocalGPUpdfSummarizer/
tmux
pixi run rpc-listener
```

Then, within two more Tmux sessions: first prepare the `llama.cpp` chat model:

```
cd LocalGPUpdfSummarizer/
tmux
pixi run serve
```

And finally, launch the summarizer on an input PDF file:

```
cd LocalGPUpdfSummarizer/
tmux
pixi run summarize INPUTFILENAME.pdf
```

Note that Tmux can be optional (usually preferred for the ability to log sessions/terminal outputs, with the added benefit of stability during longer jobs).
