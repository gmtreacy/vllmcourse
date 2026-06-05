# 1. scp the script
scp quantize.py ubuntu@<ip>:~/

# 2. ssh in
ssh ubuntu@<ip>

# 3. install guest agent for dashboard monitoring
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

# 4. install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 5. tmux
tmux new -s quant

# 6. run
uv run quantize.py

export HF_TOKEN=....

HF_REPO=gmtreacy/Qwen3-0.6B-W4A16 uv run quantize.py
