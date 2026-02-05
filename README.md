# AI Backend for SubText Project

This repository contains the AI backend components for the SubText project, primarily focusing on Natural Language Processing (NLP) tasks using the **Qwen 2.5 Instruct 7B** model and GPU-accelerated task processing.

## Project Structure

- `chat_qwen.py`: Script responsible for integrating and interacting with the **Qwen 2.5 Instruct 7B** large language model for chat-based functionalities.
- `gpu_worker.py`: The core worker script designed to poll tasks from a backend server, process them efficiently using available GPU resources, and then return the results to the backend via specified API addresses.
- `requirements.txt`: Lists all Python dependencies required for this project.
- `.gitignore`: Specifies intentionally untracked files to ignore.

## Technical Details

### Qwen 2.5 Instruct 7B Model

The project leverages the **Qwen 2.5 Instruct 7B** model, a powerful large language model developed by Alibaba Cloud. This model is fine-tuned for instruction following, making it suitable for a variety of NLP tasks including chat, question answering, and text generation. Its 7 billion parameters provide a good balance between performance and computational requirements, especially when utilizing GPU acceleration.

### GPU Worker Architecture (`gpu_worker.py`)

The `gpu_worker.py` script operates as a dedicated processing unit within the SubText AI backend. Its workflow is as follows:
1.  **Polling**: Continuously queries a designated backend server API for new processing tasks.
2.  **Processing**: Upon receiving a task, it utilizes GPU resources (managed internally, likely through libraries like PyTorch or TensorFlow) to perform the required computations, such as running inferences with the Qwen model.
3.  **Result Submission**: After processing, the worker sends the computed results back to the backend server through another specified API endpoint.
This design ensures efficient distribution and processing of AI tasks, leveraging GPU capabilities for speed.

## Setup Instructions

To get this project up and running, follow these steps:

### Prerequisites

*   Python 3.8+
*   pip (Python package installer)
*   NVIDIA GPU with appropriate drivers and CUDA toolkit for GPU acceleration.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Madcamp25W-Week4-BGM-Lab/AI
    cd AI
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate # On macOS/Linux
    pip install -r requirements.txt
    ```
    *(Note: Ensure all necessary GPU-enabled libraries like `torch` or `tensorflow-gpu` are correctly installed as part of `requirements.txt` and compatible with your CUDA setup.)*

## Usage

Once the dependencies are installed, you can run the scripts as follows:

### Running the GPU Worker

To start the GPU worker, execute the `gpu_worker.py` script. This script will continuously poll for tasks, process them, and return results. Ensure your environment variables or configuration files are set up with the correct API addresses for the backend server.

```bash
python gpu_worker.py
```

### Interacting with the Qwen Model (via `chat_qwen.py`)

The `chat_qwen.py` script provides the interface to the Qwen model. Specific usage depends on its internal implementation (e.g., whether it exposes a local API or is meant for direct execution with arguments).
