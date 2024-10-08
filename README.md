# LLM Optimization Project

This project implements a novel, generalized model optimization and inference script for `MistralForCausalLM` based LLMs. It aims to improve the performance of LLM inference in terms of latency and throughput.


## Before we begin..

- Before settling on using bitsandbytes for quantization, I experimented with two other libraries, deepspeed-fastgen and vllm. deepspeed-fastgen is the lastest
  library for inference optimization and claims to perform better than vllm. However, there are certain hardware requirements of using deepspeed-fastgen such as
  NVIDIA GPU(s) with compute capability of: 8.0, 8.6, 8.9, 9.0. (T4 has compute capability of 7.5 because of which, deepspeed-fastgen could not be testing on colab)
  CUDA toolkit 11.6+ 
  Ubuntu 20+ 
  I tried testing it on A6000 GPU (compute capability 8.6) as I have its lab access. But the script ran into errors and it has unresolved bugs which are still open
  issues on deepspeed-fastgen's github.
- Next I tried experimenting with vllm, the issue with vllm was that, it offered quantization upto 8 bit. It also offered GPU offloading but on experimenting I found
  it reduced the throughput. It has some more advanced quantization techniques like gptq and AWQ, but vllm requires the original model on huggingface to be pre-quantized
  so it can check for quant config in the config file of the huggingface model. It cannot perform in flight quantization like bitsandbytes.
- Finally I settled on bitsandbytes and the following benchmark results and script is based on that.
  
## Project Structure

- `pipeline.py`: The main script that accepts a Hugging Face model path, optimizes the model, and runs inference on user input.
- `benchmark_results.ipynb`: Jupyter notebook containing the results of testing with 32 concurrent prompts, demonstrating a throughput of 200+ tokens/sec.
- `requirements.txt`: List of Python dependencies required for this project.

## Setup

### Environment Setup

1. Ensure you have Python 3.8+ installed on your system.
2. It's recommended to use a virtual environment. Create one using:

   ```
   python -m venv llm_opt_env
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     llm_opt_env\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source llm_opt_env/bin/activate
     ```

### Installation

1. With the virtual environment activated, install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Ensure you have CUDA installed and configured for your NVIDIA GPU.

## Usage

### Running the Pipeline

1. Activate your virtual environment if it's not already activated.

2. Run the `pipeline.py` script:

   ```
   python pipeline.py
   ```

3. When prompted, enter the Hugging Face model path (e.g., `mistralai/Mistral-7B-v0.1`).

4. The script will optimize the model and perform a warmup.

5. Enter your prompt when requested.

6. The script will generate a response and display performance metrics.

### Viewing Benchmark Results

Open the `benchmark_results.ipynb` file in Jupyter Notebook or JupyterLab to view the detailed results of the benchmarking tests.

## Performance

This implementation aims to beat the following benchmarks:

- Total throughput (input + output tokens per second): 200 tokens/sec
- Input tokens: 128
- Output tokens: 128
- Concurrency: 32
- GPU: 1 x NVIDIA Tesla T4 (16GB VRAM)
- Model dtype: Any dtype supported by the GPU

## Notes

- Ensure you have sufficient GPU memory (16GB+) to run the models.
- The script is optimized for the specified GPU (NVIDIA Tesla T4), but may work on other CUDA-capable GPUs with sufficient memory.
- Performance may vary based on the specific model used and the complexity of the input prompts.

 

