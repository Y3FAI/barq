apt update && apt upgrade -y

# 1. Update the system

apt update && apt upgrade -y

# 2. Install Python, Pip, and Git

apt install -y python3-pip python3-venv git

# 3. Create a folder for your app

mkdir barq_app
cd barq_app

# 4. Create a virtual environment (keeps things clean)

python3 -m venv venv
source venv/bin/activate

# Install the server version (CPU optimized)

pip install "llama-cpp-python[server]"

# Download the GGUF file directly

wget https://huggingface.co/y3fai/barq/resolve/main/qwen3-1.7b.Q4_K_M.gguf

python3 -m llama_cpp.server --model ./qwen3-1.7b.Q4_K_M.gguf --host 0.0.0.0 --port 8000
