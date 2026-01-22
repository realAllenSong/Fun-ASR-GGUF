"""
端到端 ASR 推理脚本 (End-to-End ASR Inference)

流程:
1. 加载音频文件 (MP3/WAV/etc) -> PCM 16kHz
2. ONNX Encoder: 音频 -> Audio Embedding
3. 从 GGUF 读取 token_embd.weight，生成 prefix/suffix embedding
4. 拼接 [prefix + audio + suffix]
5. llama.dll 直接注入 embedding 并生成文本

依赖:
- onnxruntime
- pydub (音频处理)
- gguf (读取 GGUF 模型权重)
- llama.dll / ggml.dll (推理) 

llama.cpp 下载地址：
https://github.com/ggml-org/llama.cpp/releases/download/b7786/llama-b7786-bin-win-vulkan-x64.zip

"""

import sys
import os
import ctypes
import numpy as np
import time

# Modules
import nano_llama # Import entire module to access initialized globals
from nano_ctc import load_ctc_tokens, decode_ctc, align_timestamps
from nano_audio import load_audio
from nano_onnx import load_onnx_models, encode_audio

from hotword.hot_phoneme import PhonemeCorrector




# =========================================================================
# 音频文件
# =========================================================================

# 输入音频
INPUT_AUDIO         = "input.mp3"

# 上下文信息 (可选，留空则不使用)
CONTEXT             = "这是1004期睡前消息节目，主持人叫督工，助理叫静静"

# 是否启用 CTC 辅助 (开启后可提供时间戳和热词回忆，关闭则仅使用 LLM)
ENABLE_CTC          = False

# 语言设置
# 中文、英文、日文 for Fun-ASR-Nano-2512
# 中文、英文、粤语、日文、韩文、越南语、印尼语、泰语、马来语、菲律宾语、阿拉伯语、
# 印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、
# 匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、
# 斯洛伐克语、斯洛文尼亚语、瑞典语 for Fun-ASR-MLT-Nano-2512
LANGUAGE = None


# =========================================================================
# Vulkan 选项
# =========================================================================

# os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan
# os.environ["GGML_VK_VISIBLE_DEVICES"] = "0"   # 禁止 Vulkan 用独显（强制用集显）
# os.environ["GGML_VK_DISABLE_F16"] = "1"       # 禁止 VulkanFP16 计算（Intel集显fp16有溢出问题）


# =========================================================================
# 系统配置
# =========================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径
MODEL_DIR           = os.path.join(SCRIPT_DIR, "model-gguf")
ENCODER_ONNX_PATH   = os.path.join(MODEL_DIR, "Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx")  
CTC_ONNX_PATH       = os.path.join(MODEL_DIR, "Fun-ASR-Nano-CTC.int8.onnx")
DECODER_GGUF_PATH   = os.path.join(MODEL_DIR, "Fun-ASR-Nano-Decoder.q8_0.gguf")
TOKENS_PATH         = os.path.join(MODEL_DIR, "tokens.txt")

# llama.cpp DLL 目录
BIN_DIR             = os.path.join(SCRIPT_DIR, "bin")

# 音频参数
SAMPLE_RATE = 16000

# 推理参数
core_num = os.cpu_count()
N_PREDICT = 512                 # 最大生成 token 数
N_THREADS = core_num // 2       # 生成阶段，内存带宽受限，使用物理核心数
N_THREADS_BATCH = core_num      # 读取阶段，计算密集，线程全开
QUIET_MODE = True               # 静默模式 (关闭 llama.cpp 调试信息)
INJECT_CHUNK_SIZE = 512         # 注入 Embedding 时的分块大小 
N_UBATCH = 512                  # llama.cpp 内部物理 batch 大小
STOP_TOKENS = [151643, 151645]



# =========================================================================
# Helper Functions in Main
# =========================================================================

def load_gguf_model_wrapper():
    """加载 GGUF LLM 解码器"""
    print(f"\n[2] 加载 GGUF LLM Decoder")
    t_start = time.perf_counter()
    
    # Initialize Llama Library
    nano_llama.init_llama_lib(BIN_DIR)
    
    model_params = nano_llama.llama_model_default_params()
    model = nano_llama.llama_model_load_from_file(DECODER_GGUF_PATH.encode('utf-8'), model_params)
    
    t_cost = time.perf_counter() - t_start
    if not model:
        print("    ERROR: Failed to load model")
        return None, None, None, None, t_cost
        
    print(f"    Decoder: {os.path.basename(DECODER_GGUF_PATH)} (耗时: {t_cost:.2f}s)")
    
    vocab = nano_llama.llama_model_get_vocab(model)
    eos_token = nano_llama.llama_vocab_eos(vocab)
    
    return model, vocab, eos_token, t_cost

def load_embedding_weights():
    """加载 token embedding 权重"""
    print("\n[3] 加载 token embedding 权重...")
    t_start = time.perf_counter()
    
    embedding_table = nano_llama.get_token_embeddings_gguf(DECODER_GGUF_PATH)
    if embedding_table is None:
        return None, 0
    
    t_read = time.perf_counter() - t_start
    print(f"    Embedding table: {embedding_table.shape} (耗时: {t_read*1000:.2f}ms)")
    return embedding_table, t_read

def load_hotword_corrector():
    """加载热词纠错器"""
    print("\n[4] 初始化热词纠错器...")
    corrector = PhonemeCorrector(threshold=0.8)
    
    hot_path = os.path.join(SCRIPT_DIR, "hot.txt")
    loaded_count = 0
    
    if os.path.exists(hot_path):
        content = open(hot_path, "r", encoding="utf-8").read()
        loaded_count = corrector.update_hotwords(content)
        print(f"    已加载热词文件: {hot_path} ({loaded_count} 条)")
    else:
        print(f"    未找到 hot.txt")
        corrector.update_hotwords("")
        
    return corrector, loaded_count

def process_audio_file(audio_path, encoder_sess):
    """加载并编码音频"""
    print(f"\n[1] 加载音频和 Encode: {os.path.basename(audio_path)}")
    
    audio = load_audio(audio_path, SAMPLE_RATE)
    audio_len = len(audio)
    print(f"    音频长度: {audio_len} samples ({audio_len/SAMPLE_RATE:.2f}s)")
    
    t_start = time.perf_counter()
    audio_embd, enc_output = encode_audio(audio, encoder_sess)
    t_cost = time.perf_counter() - t_start
    
    print(f"    耗时: {t_cost*1000:.2f}ms")
    return audio_embd, enc_output, audio_len, t_cost

def run_ctc_pass(ctc_sess, enc_output, ctc_id2token, corrector):
    """CTC Decode & Hotword Matching"""
    print("\n[2] CTC Decode")
    t_start = time.perf_counter()
    
    ctc_logits = ctc_sess.run(None, {"enc_output": enc_output})[0]
    ctc_text, ctc_results = decode_ctc(ctc_logits, ctc_id2token)
    
    hotwords = []
    if corrector and corrector.hotwords and ctc_text:
        res = corrector.correct(ctc_text, k=10)
        
        candidates = set()
        for _, hw, _ in res.matchs:
            candidates.add(hw)
        for _, hw, _ in res.similars:
            candidates.add(hw)
        hotwords = list(candidates)
    
    t_cost = time.perf_counter() - t_start
    
    print(f"    CTC 识别结果：{ctc_text}")
    
    # 格式化时间戳输出
    ts_list = [float(f"{r.start:.2f}") for r in ctc_results]
    print(f"    CTC 时间戳：{ts_list[:10]} ......")
    
    if hotwords:
        print(f"    检索到热词：{hotwords}")
    else:
        print(f"    未检索到热词")
    
    return ctc_results, hotwords, t_cost

def prepare_prompt_embeddings(vocab, embedding_table, matched_hotwords=None, context=None, language=LANGUAGE):
    """生成 Prompt"""
    print(f"\n[3] 生成 Prompt (语言: {language})")
    
    PREFIX_PROMPT = "<|im_start|>system\n"
    PREFIX_PROMPT += "You are a helpful assistant."
    PREFIX_PROMPT += "<|im_end|>\n<|im_start|>user\n"

    if matched_hotwords or context:
        
        if context:
            PREFIX_PROMPT += f"请结合上下文信息，更加准确地完成语音转写任务。\n\n\n"
            PREFIX_PROMPT += f"**上下文信息：**{context}\n\n\n"
            
        if matched_hotwords:
            hotwords = ", ".join(matched_hotwords)
            PREFIX_PROMPT += f"热词列表：[{hotwords}]\n"
    
    if language is None:
        PREFIX_PROMPT += "语音转写："
    else:
        PREFIX_PROMPT += f"语音转写成{language}："

    # print( '-'*20 + '\n' + PREFIX_PROMPT + '\n' + '-'*20 + '\n')
    SUFFIX_PROMPT = "<|im_end|>\n<|im_start|>assistant\n"
    
    prefix_tokens = nano_llama.text_to_tokens(vocab, PREFIX_PROMPT)
    suffix_tokens = nano_llama.text_to_tokens(vocab, SUFFIX_PROMPT)
    
    prefix_embd = embedding_table[prefix_tokens].astype(np.float32)
    suffix_embd = embedding_table[suffix_tokens].astype(np.float32)
    
    print(f"    Prefix: {len(prefix_tokens)} tokens")
    print(f"    Suffix: {len(suffix_tokens)} tokens")
    
    return prefix_embd, suffix_embd, len(prefix_tokens), len(suffix_tokens)

def setup_inference_context(model, full_embd, n_tokens_input):
    """创建上下文并注入 Embedding"""
    print(f"\n[4] 注入 embeddings ({n_tokens_input} tokens)...")
    
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 2048
    ctx_params.n_batch = 2048
    ctx_params.n_ubatch = N_UBATCH
    ctx_params.embeddings = False
    ctx_params.no_perf = True
    ctx_params.n_threads = N_THREADS
    ctx_params.n_threads_batch = N_THREADS_BATCH
    
    ctx = nano_llama.llama_init_from_model(model, ctx_params)
    if not ctx:
        return None, 0
    
    t_start = time.perf_counter()
    batch_embd = nano_llama.llama_batch_init(n_tokens_input, full_embd.shape[1], 1)
    
    # Prepare batch
    batch_embd.n_tokens = n_tokens_input
    # 注意: batch_embd.token 在 C 中是指针，这里设置为 None 或正确的类型
    # 在 之前的代码中：batch_embd.token = ctypes.cast(None, ctypes.POINTER(llama_token))
    # 这里需要确保 nano_llama.llama_token 可用
    batch_embd.token = ctypes.cast(None, ctypes.POINTER(nano_llama.llama_token))
    
    if not full_embd.flags['C_CONTIGUOUS']:
        full_embd = np.ascontiguousarray(full_embd)
    ctypes.memmove(batch_embd.embd, full_embd.ctypes.data, full_embd.nbytes)
    
    for k in range(n_tokens_input):
        batch_embd.pos[k] = k
        batch_embd.n_seq_id[k] = 1
        batch_embd.seq_id[k][0] = 0
        batch_embd.logits[k] = 1 if k == n_tokens_input - 1 else 0
        
    ret = nano_llama.llama_decode(ctx, batch_embd)
    nano_llama.llama_batch_free(batch_embd)
    
    if ret != 0:
        print(f"    ERROR: Decode failed (ret={ret})")
        nano_llama.llama_free(ctx)
        return None, 0

    t_cost = time.perf_counter() - t_start
    print(f"    注入耗时: {t_cost*1000:.2f}ms")
    return ctx, t_cost

def run_generation(ctx, vocab, eos_token, n_input_tokens):
    """生成文本"""
    print(f"\n[5] 生成文本 (最大 {N_PREDICT} tokens)...")
    print("=" * 70)
    
    vocab_size = nano_llama.llama_vocab_n_tokens(vocab)
    batch_text = nano_llama.llama_batch_init(1, 0, 1)
    batch_text.n_tokens = 1
    
    generated_text = ""
    current_pos = n_input_tokens
    tokens_generated = 0
    decoder = nano_llama.ByteDecoder()
    
    t_gen_start = time.perf_counter()
    
    try:
        for _ in range(N_PREDICT):
            logits_ptr = nano_llama.llama_get_logits(ctx)
            # 注意: 这里简单假设 vocab_size 足够大，直接读取
            logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
            token_id = int(np.argmax(logits_arr))
            
            if token_id == eos_token or token_id in STOP_TOKENS:
                break
            
            raw_bytes = nano_llama.token_to_bytes(vocab, token_id)
            text_piece = decoder.decode(raw_bytes)
            
            print(text_piece, end="", flush=True)
            generated_text += text_piece
            tokens_generated += 1
            
            batch_text.token[0] = token_id
            batch_text.pos[0] = current_pos
            batch_text.n_seq_id[0] = 1
            batch_text.seq_id[0][0] = 0
            batch_text.logits[0] = 1
            
            if nano_llama.llama_decode(ctx, batch_text) != 0:
                break
            
            current_pos += 1
            
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    remaining = decoder.flush()
    if remaining:
        print(remaining, end="", flush=True)
        generated_text += remaining
    
    t_cost = time.perf_counter() - t_gen_start
    print("\n" + "=" * 70)
    
    nano_llama.llama_batch_free(batch_text)
    return generated_text, tokens_generated, t_cost

# =========================================================================
# 主函数 (Main)
# =========================================================================

def main():
    print("=" * 70)
    print("SenseVoice Hybrid ASR 推理 (CTC + LLM)")
    print("=" * 70)
    
    # 加载 ONNX Models
    encoder_sess, ctc_sess, t_load_onnx = load_onnx_models(ENCODER_ONNX_PATH, CTC_ONNX_PATH)
    
    # 加载GGUF LLM Decoder
    model, vocab, eos_token, t_load_dec = load_gguf_model_wrapper()
    if not model: return 1
    
    # 加载 Embedding 权重
    embedding_table, t_load_embd = load_embedding_weights()
    if embedding_table is None: return 1
    
    # 加载 CTC 词表
    ctc_id2token = load_ctc_tokens(TOKENS_PATH)

    # 初始化热词纠错器
    corrector, loaded_count = load_hotword_corrector()

    print("\n" + "=" * 70)
    print("模型加载完成，准备处理音频...")
    print("=" * 70)
    
    # 音频编码
    audio_embd, enc_output, audio_len, t_encode_audio = process_audio_file(INPUT_AUDIO, encoder_sess)
    
    # CTC 解码
    if ENABLE_CTC:
        ctc_results, hotwords, t_ctc_cost = run_ctc_pass(ctc_sess, enc_output, ctc_id2token, corrector)
    else:
        print("\n[2] CTC Decode (Skipped)")
        ctc_results = []
        hotwords = []
        t_ctc_cost = 0.0
    
    # 准备提示词
    prefix_embd, suffix_embd, n_prefix, n_suffix = prepare_prompt_embeddings(vocab, embedding_table, hotwords, CONTEXT, LANGUAGE)
    
    # 拼接 embd
    full_embd = np.concatenate([prefix_embd, audio_embd.astype(np.float32), suffix_embd], axis=0)
    n_input_tokens = full_embd.shape[0]
    
    # 注入 embd
    ctx, t_inject = setup_inference_context(model, full_embd, n_input_tokens)
    if not ctx: return 1
    
    # LLM 解码
    text, n_gen, t_gen = run_generation(ctx, vocab, eos_token, n_input_tokens)

    # 时间戳对齐
    t_align_start = time.perf_counter()
    aligned_result = align_timestamps(ctc_results, text)
    t_align = time.perf_counter() - t_align_start
    
    # 打印对齐结果
    print(f"    对齐耗时: {t_align*1000:.2f}ms")
    print(f"    最终结果 (前50字符):")
    for r in aligned_result[:10]:
        print(f"      {r['start']:.2f}-{r['end']:.2f}: {r['char']}")
    if len(aligned_result) > 10: print("      ...")
    
    # 统计
    tps_out = n_gen / t_gen if t_gen > 0 else 0
    tps_in = n_input_tokens / t_inject if t_inject > 0 else 0
    t_total = t_encode_audio + t_ctc_cost + t_inject + t_gen
    
    print(f"\n[统计]")
    print(f"  音频长度: {audio_len/SAMPLE_RATE:.2f}s")
    print(f"  Decoder输入: {tps_in:5.0f} tokens/s (all: {n_input_tokens}, prefix:{n_prefix}, audio:{audio_embd.shape[0]}, suffix:{n_suffix})")
    print(f"  Decoder输出: {tps_out:5.0f} tokens/s (all: {n_gen})")

    print(f"\n[加载耗时]")
    print(f"  - ONNX加载： {t_load_onnx*1000:5.0f}ms")
    print(f"  - GGUF加载： {t_load_dec*1000:5.0f}ms")
    print(f"  - Embd读取： {t_load_embd*1000:5.0f}ms")
    
    print(f"\n[转录耗时]")
    print(f"  - 音频编码： {t_encode_audio*1000:5.0f}ms")
    print(f"  - CTC解码：  {t_ctc_cost*1000:5.0f}ms")
    print(f"  - LLM读取：  {t_inject*1000:5.0f}ms")
    print(f"  - LLM生成：  {t_gen*1000:5.0f}ms")
    print(f"  - 总耗时：   {t_total:5.2f}s")
    
    # 清理
    nano_llama.llama_free(ctx)
    nano_llama.llama_model_free(model)
    nano_llama.llama_backend_free()
    
    print("\n[完成]")
    return 0

if __name__ == "__main__":
    exit(main())
