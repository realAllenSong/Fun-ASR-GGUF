
import torch
import os
import json
import shutil
import subprocess
import sys

# =========================================================================
# 配置部分
# =========================================================================

# 源模型路径
SOURCE_MODEL_PATH = './Fun-ASR-Nano-2512'
CONFIG_PATH = f'{SOURCE_MODEL_PATH}/Qwen3-0.6B/config.json'

# 中间产物 (HF 格式) 输出路径
OUTPUT_HF_DIR = './model-gguf/Qwen3-0.6B'
# Tokenizer 输出路径
OUTPUT_TOKENIZER_DIR = './model-gguf/Qwen3-0.6B'

# 最终 GGUF 输出文件 (Int8/Q8_0)
OUTPUT_GGUF_FILE = './model-gguf/qwen3-0.6b-asr-q8_0.gguf'

# Llama.cpp 路径 (自动寻找 convert_hf_to_gguf.py)
# 优先尝试 llama.cpp 目录，如果不存在尝试 llama.cpp-master
if os.path.exists('./llama.cpp/convert_hf_to_gguf.py'):
    LLAMA_CPP_PATH = './llama.cpp'
elif os.path.exists('./llama.cpp-master/convert_hf_to_gguf.py'):
    LLAMA_CPP_PATH = './llama.cpp-master'
else:
    # Fallback to current directory or error later
    LLAMA_CPP_PATH = './llama.cpp' 

CONVERT_SCRIPT = f'{LLAMA_CPP_PATH}/convert_hf_to_gguf.py'


def main():
    # ---------------------------------------------------------------------
    # 1. 提取 LLM 并保存为 Hugging Face 格式 (如果已存在则跳过，为了节省时间)
    # ---------------------------------------------------------------------
    print("\n[Stage 1] Checking/Extracting LLM Decoder to Hugging Face format...")
    
    # 简单的检查，如果 config.json 和 safetensors 存在，假设已经解压过了，可以选择跳过
    # 为了保险起见，这里还是执行一遍提取逻辑，但你可以手动注释掉如果这步耗时太长
    
    # 尝试导入 Qwen3 类
    try:
        from transformers import Qwen3ForCausalLM, Qwen3Config
    except ImportError:
        try:
            from transformers import Qwen2ForCausalLM as Qwen3ForCausalLM
            from transformers import Qwen2Config as Qwen3Config
        except ImportError:
            from transformers import AutoModelForCausalLM as Qwen3ForCausalLM
            from transformers import AutoConfig as Qwen3Config

    # 加载完整 PyTorch 模型 (FunASR 格式)
    model_pt_path = f'{SOURCE_MODEL_PATH}/model.pt'
    
    if os.path.exists(os.path.join(OUTPUT_HF_DIR, "model.safetensors")):
         print(f"HF model appears to exist in {OUTPUT_HF_DIR}. Skipping extraction.")
    else:
        print(f"Loading full model from {model_pt_path} ...")
        full_model = torch.load(model_pt_path, map_location='cpu')

        # 提取 LLM 权重
        llm_weights = {}
        print("Extracting LLM weights...")
        for key in full_model.keys():
            if key.startswith('llm.'):
                # 将键名从 llm.model.xxx 转换为 model.xxx (HF 标准格式)
                hf_key = key.replace('llm.', '')
                llm_weights[hf_key] = full_model[key]
        
        print(f"Extracted {len(llm_weights)} weight keys.")
        del full_model
        
        # 加载配置
        print(f"Loading config from {CONFIG_PATH} ...")
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
        config = Qwen3Config(**config_dict)
        
        # 初始化空模型
        print("Initializing empty Qwen3ForCausalLM...")
        qwen_model = Qwen3ForCausalLM(config)

        # 加载权重
        print("Loading state dict into LLM...")
        qwen_model.load_state_dict(llm_weights, strict=True)
        
        # 保存 HF 模型 (Safetensors)
        os.makedirs(OUTPUT_HF_DIR, exist_ok=True)
        print(f"Saving HF model to {OUTPUT_HF_DIR} ...")
        qwen_model.save_pretrained(OUTPUT_HF_DIR, safe_serialization=True)
        
        # 复制 tokenizer 文件到单独目录
        print(f"Copying tokenizer files to {OUTPUT_TOKENIZER_DIR} ...")
        os.makedirs(OUTPUT_TOKENIZER_DIR, exist_ok=True)
        original_tokenizer_dir = f'{SOURCE_MODEL_PATH}/Qwen3-0.6B'
        files_to_copy = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'generation_config.json']
        for file in files_to_copy:
            src = os.path.join(original_tokenizer_dir, file)
            dst = os.path.join(OUTPUT_TOKENIZER_DIR, file)
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"  Copied {file}")
        
        print("HF Model and Tokenizer saved successfully.")

    # ---------------------------------------------------------------------
    # 2. 转换为 GGUF 格式 (Int8)
    # ---------------------------------------------------------------------
    print("\n[Stage 2] Converting HF model to GGUF (Int8 - q8_0)...")
    
    if not os.path.exists(CONVERT_SCRIPT):
        print(f"Error: Llama.cpp conversion script not found at {CONVERT_SCRIPT}")
        return

    # 构建命令
    # python convert.py model_dir --outfile ... --outtype q8_0
    
    cmd = [
        sys.executable,
        CONVERT_SCRIPT,
        OUTPUT_HF_DIR,
        '--outfile', OUTPUT_GGUF_FILE,
        '--outtype', 'q8_0',  # <-- Key change here
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ GGUF Int8 conversion successful! Output: {OUTPUT_GGUF_FILE}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ GGUF conversion failed with error: {e}")

if __name__ == "__main__":
    main()
