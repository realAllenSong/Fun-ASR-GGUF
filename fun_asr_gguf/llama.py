import sys
import os
import ctypes
import numpy as np
import gguf
from pathlib import Path
from os.path import relpath
from . import logger

# =========================================================================
# Configuration
# =========================================================================
# QUIET_LOGS = True 时，不打印任何日志。但现在我们路由到 logger。
QUIET_LOGS = False
_log_callback_ref = None

# =========================================================================
# Type Definitions
# =========================================================================

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.POINTER(ctypes.c_void_p)),
        ("tensor_buft_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]

class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.POINTER(ctypes.c_void_p)),
        ("n_samplers", ctypes.c_size_t),
    ]

class llama_sampler_chain_params(ctypes.Structure):
    _fields_ = [
        ("no_perf", ctypes.c_bool),
    ]

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]

# =========================================================================
# Llama Library Bindings
# =========================================================================

# Global library references
llama = None
ggml = None
ggml_base = None

# Global function pointers
llama_log_set = None
llama_backend_init = None
llama_backend_free = None
llama_model_default_params = None
llama_model_load_from_file = None
llama_model_free = None
llama_model_get_vocab = None
llama_context_default_params = None
llama_init_from_model = None
llama_free = None
llama_batch_init = None
llama_batch_free = None
llama_decode = None
llama_get_logits = None
llama_tokenize = None
llama_vocab_n_tokens = None
llama_vocab_eos = None
llama_token_to_piece = None
llama_get_memory = None
llama_memory_clear = None
llama_model_n_embd = None

# Sampler
llama_sampler_chain_default_params = None
llama_sampler_chain_init = None
llama_sampler_chain_add = None
llama_sampler_init_greedy = None
llama_sampler_init_dist = None
llama_sampler_init_temp = None
llama_sampler_init_top_k = None
llama_sampler_init_top_p = None
llama_sampler_sample = None
llama_sampler_free = None

def init_llama_lib():
    """初始化 llama.cpp 库，支持跨平台加载"""
    global llama, ggml, ggml_base
    global llama_log_set, llama_backend_init, llama_backend_free
    global llama_model_default_params, llama_model_load_from_file, llama_model_free, llama_model_get_vocab
    global llama_context_default_params, llama_init_from_model, llama_free
    global llama_batch_init, llama_batch_free
    global llama_decode, llama_get_logits, llama_tokenize
    global llama_get_memory, llama_memory_clear, llama_model_n_embd
    global llama_vocab_n_tokens, llama_vocab_eos, llama_token_to_piece
    global llama_sampler_chain_default_params, llama_sampler_chain_init, llama_sampler_chain_add
    global llama_sampler_init_greedy, llama_sampler_init_dist, llama_sampler_init_temp
    global llama_sampler_init_top_k, llama_sampler_init_top_p, llama_sampler_sample, llama_sampler_free
    global _log_callback_ref

    if llama is not None:
        return

    # 获取库文件所在目录 (模块目录下的 bin)
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")

    # DLL 命名处理
    if sys.platform == "win32":
        GGML_DLL = "ggml.dll"
        GGML_BASE_DLL = "ggml-base.dll"
        LLAMA_DLL = "llama.dll"
    elif sys.platform == "darwin":
        GGML_DLL = "libggml.dylib"
        GGML_BASE_DLL = "libggml-base.dylib"
        LLAMA_DLL = "libllama.dylib"
    else:
        GGML_DLL = "libggml.so"
        GGML_BASE_DLL = "libggml-base.so"
        LLAMA_DLL = "libllama.so"

    original_cwd = os.getcwd()
    os.chdir(lib_dir)
    
    # Windows DLL directory treatment
    if sys.platform == "win32" and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(lib_dir)

    try:
        ggml = ctypes.CDLL(os.path.join(lib_dir, GGML_DLL))
        ggml_base = ctypes.CDLL(os.path.join(lib_dir, GGML_BASE_DLL))
        llama = ctypes.CDLL(os.path.join(lib_dir, LLAMA_DLL))

        # 设置日志回调
        LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
        llama_log_set = llama.llama_log_set
        llama_log_set.argtypes = [LOG_CALLBACK, ctypes.c_void_p]
        llama_log_set.restype = None
        
        # 默认开启日志路由
        configure_logging(quiet=QUIET_LOGS)

        # 加载后端
        ggml_backend_load_all = ggml.ggml_backend_load_all
        ggml_backend_load_all.argtypes = []
        ggml_backend_load_all.restype = None
        ggml_backend_load_all()

        llama_backend_init = llama.llama_backend_init
        llama_backend_init.argtypes = []
        llama_backend_init.restype = None
        llama_backend_init()

    finally:
        os.chdir(original_cwd)

    # 绑定其他函数
    llama_backend_free = llama.llama_backend_free
    llama_backend_free.argtypes = []
    llama_backend_free.restype = None

    llama_model_default_params = llama.llama_model_default_params
    llama_model_default_params.argtypes = []
    llama_model_default_params.restype = llama_model_params

    llama_model_load_from_file = llama.llama_model_load_from_file
    llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
    llama_model_load_from_file.restype = ctypes.c_void_p

    llama_model_free = llama.llama_model_free
    llama_model_free.argtypes = [ctypes.c_void_p]
    llama_model_free.restype = None

    llama_model_get_vocab = llama.llama_model_get_vocab
    llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    llama_model_get_vocab.restype = ctypes.c_void_p

    llama_model_n_embd = llama.llama_model_n_embd
    llama_model_n_embd.argtypes = [ctypes.c_void_p]
    llama_model_n_embd.restype = ctypes.c_int32

    # Context
    llama_context_default_params = llama.llama_context_default_params
    llama_context_default_params.argtypes = []
    llama_context_default_params.restype = llama_context_params

    llama_init_from_model = llama.llama_init_from_model
    llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
    llama_init_from_model.restype = ctypes.c_void_p

    llama_free = llama.llama_free
    llama_free.argtypes = [ctypes.c_void_p]
    llama_free.restype = None

    # Batch
    llama_batch_init = llama.llama_batch_init
    llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    llama_batch_init.restype = llama_batch

    llama_batch_free = llama.llama_batch_free
    llama_batch_free.argtypes = [llama_batch]
    llama_batch_free.restype = None

    # Decode
    llama_decode = llama.llama_decode
    llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
    llama_decode.restype = ctypes.c_int32

    # Logits
    llama_get_logits = llama.llama_get_logits
    llama_get_logits.argtypes = [ctypes.c_void_p]
    llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    # Tokenize
    llama_tokenize = llama.llama_tokenize
    llama_tokenize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(llama_token), ctypes.c_int32,
        ctypes.c_bool, ctypes.c_bool,
    ]
    llama_tokenize.restype = ctypes.c_int32

    # Vocab
    llama_vocab_n_tokens = llama.llama_vocab_n_tokens
    llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    llama_vocab_n_tokens.restype = ctypes.c_int32

    llama_vocab_eos = llama.llama_vocab_eos
    llama_vocab_eos.argtypes = [ctypes.c_void_p]
    llama_vocab_eos.restype = llama_token

    llama_token_to_piece = llama.llama_token_to_piece
    llama_token_to_piece.argtypes = [ctypes.c_void_p, llama_token, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool]
    llama_token_to_piece.restype = ctypes.c_int

    # Memory (KV Cache)
    llama_get_memory = llama.llama_get_memory
    llama_get_memory.argtypes = [ctypes.c_void_p]
    llama_get_memory.restype = ctypes.c_void_p

    llama_memory_clear = llama.llama_memory_clear
    llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    llama_memory_clear.restype = None

    # Sampler
    try:
        llama_sampler_chain_default_params = llama.llama_sampler_chain_default_params
        llama_sampler_chain_default_params.argtypes = []
        llama_sampler_chain_default_params.restype = llama_sampler_chain_params

        llama_sampler_chain_init = llama.llama_sampler_chain_init
        llama_sampler_chain_init.argtypes = [llama_sampler_chain_params]
        llama_sampler_chain_init.restype = ctypes.c_void_p

        llama_sampler_chain_add = llama.llama_sampler_chain_add
        llama_sampler_chain_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        llama_sampler_chain_add.restype = None

        llama_sampler_init_greedy = llama.llama_sampler_init_greedy
        llama_sampler_init_greedy.argtypes = []
        llama_sampler_init_greedy.restype = ctypes.c_void_p

        llama_sampler_init_dist = llama.llama_sampler_init_dist
        llama_sampler_init_dist.argtypes = [ctypes.c_uint32]
        llama_sampler_init_dist.restype = ctypes.c_void_p

        llama_sampler_init_temp = llama.llama_sampler_init_temp
        llama_sampler_init_temp.argtypes = [ctypes.c_float]
        llama_sampler_init_temp.restype = ctypes.c_void_p

        llama_sampler_init_top_k = llama.llama_sampler_init_top_k
        llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
        llama_sampler_init_top_k.restype = ctypes.c_void_p

        llama_sampler_init_top_p = llama.llama_sampler_init_top_p
        llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
        llama_sampler_init_top_p.restype = ctypes.c_void_p

        llama_sampler_sample = llama.llama_sampler_sample
        llama_sampler_sample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        llama_sampler_sample.restype = llama_token

        llama_sampler_free = llama.llama_sampler_free
        llama_sampler_free.argtypes = [ctypes.c_void_p]
        llama_sampler_free.restype = None
    except AttributeError:
        # 版本较旧的 llama.cpp 可能没有这些导出
        logger.warning("llama.cpp 库中缺少原生采样 API，将无法使用原生采样优化。")

def load_model(model_path: str, n_gpu_layers: int = -1):
    """加载 GGUF 模型（包含环境优化和错误排查日志）"""
    init_llama_lib()
    
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
    model_path_obj = Path(model_path).resolve()
    
    # 计算相对于 bin 的路径，有些情况下 dll 寻找模型用相对路径更稳
    try:
        model_rel = relpath(model_path_obj, lib_dir)
    except ValueError:
        model_rel = model_path_obj.as_posix()

    model_params = llama_model_default_params()
    if n_gpu_layers != -1:
        model_params.n_gpu_layers = n_gpu_layers

    # 如果在 Windows 上，临时切目录以确保模型加载器能找到一些奇怪的路径
    original_cwd = os.getcwd()
    os.chdir(lib_dir)
    try:
        model = llama_model_load_from_file(
            model_rel.encode('utf-8'),
            model_params
        )
    finally:
        os.chdir(original_cwd)

    if not model:
        logger.error(f"模型加载失败: {model_path_obj}")
        logger.error(f"当前工作目录: {original_cwd}")
        logger.error(f"模型是否存在: {model_path_obj.exists()}")
        return None
        
    return model

def create_context(model, n_ctx=2048, n_batch=2048, n_ubatch=512, n_seq_max=1, 
                   embeddings=False, pooling_type=0, flash_attn=True, 
                   offload_kqv=True, no_perf=True, n_threads=None):
    """创建 ASR 专用的上下文"""
    params = llama_context_default_params()
    params.n_ctx = n_ctx
    params.n_batch = n_batch
    params.n_ubatch = n_ubatch
    params.n_seq_max = n_seq_max
    params.embeddings = embeddings
    params.pooling_type = pooling_type
    params.flash_attn_type = 1 if flash_attn else 0  # 1 = ON, 0 = OFF (auto typically uses what's available)
    params.offload_kqv = offload_kqv
    params.no_perf = no_perf
    
    if n_threads:
        params.n_threads = n_threads
        params.n_threads_batch = n_threads
    else:
        params.n_threads = os.cpu_count() // 2
        params.n_threads_batch = os.cpu_count()

    return llama_init_from_model(model, params)

def create_batch(n_tokens, embd_dim=0, n_seq_max=1):
    """创建推理用的 Batch"""
    return llama_batch_init(n_tokens, embd_dim, n_seq_max)

class LlamaSampler:
    """采样器的面向对象封装"""
    def __init__(self, ptr):
        self.ptr = ptr

    def sample(self, ctx, idx=-1):
        """采样一个 Token"""
        return llama_sampler_sample(self.ptr, ctx, idx)

    def free(self):
        """释放采样器资源"""
        if self.ptr:
            llama_sampler_free(self.ptr)
            self.ptr = None

def create_sampler(temperature=0.8, top_k=50, top_p=1.0, seed=None):
    """创建 ASR 专用的采样器对象"""
    import time
    if seed is None:
        seed = int(time.time())
        
    sparams = llama_sampler_chain_default_params()
    smpl_ptr = llama_sampler_chain_init(sparams)
    
    if temperature > 0:
        llama_sampler_chain_add(smpl_ptr, llama_sampler_init_top_k(top_k))
        llama_sampler_chain_add(smpl_ptr, llama_sampler_init_top_p(top_p, 1))
        llama_sampler_chain_add(smpl_ptr, llama_sampler_init_temp(temperature))
        llama_sampler_chain_add(smpl_ptr, llama_sampler_init_dist(seed))
    else:
        llama_sampler_chain_add(smpl_ptr, llama_sampler_init_greedy())
        
    return LlamaSampler(smpl_ptr)

# =========================================================================
# 日志回调
# =========================================================================

def python_log_callback(level, message, user_data):
    if not message: return
    try:
        msg_str = message.decode('utf-8', errors='replace').strip()
        if not msg_str or msg_str in ['.', '\n']: return
        
        if level == 2: logger.error(f"[llama.cpp] {msg_str}")
        elif level == 3: logger.warning(f"[llama.cpp] {msg_str}")
        else: logger.info(f"[llama.cpp] {msg_str}")
    except Exception: pass

def configure_logging(quiet=False):
    global _log_callback_ref
    if not llama_log_set: return
    
    LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    if not quiet:
        _log_callback_ref = LOG_CALLBACK(python_log_callback)
        llama_log_set(_log_callback_ref, None)
    else:
        _log_callback_ref = LOG_CALLBACK(lambda l, m, u: None)
        llama_log_set(_log_callback_ref, None)

# =========================================================================
# Utilities
# =========================================================================

class ByteDecoder:
    def __init__(self):
        self.buffer = b""
    
    def decode(self, raw_bytes):
        self.buffer += raw_bytes
        result = ""
        while self.buffer:
            try:
                result += self.buffer.decode('utf-8')
                self.buffer = b""
                break
            except UnicodeDecodeError as e:
                if e.reason == 'unexpected end of data' or 'invalid continuation' in e.reason:
                    if e.start > 0:
                        result += self.buffer[:e.start].decode('utf-8', errors='replace')
                        self.buffer = self.buffer[e.start:]
                    break
                else:
                    result += self.buffer[:1].decode('utf-8', errors='replace')
                    self.buffer = self.buffer[1:]
        return result
    
    def flush(self):
        if self.buffer:
            result = self.buffer.decode('utf-8', errors='replace')
            self.buffer = b""
            return result
        return ""

def text_to_tokens(vocab, text):
    text_bytes = text.encode("utf-8")
    n_tokens_max = len(text_bytes) + 32
    tokens = (llama_token * n_tokens_max)()
    n = llama_tokenize(vocab, text_bytes, len(text_bytes), tokens, n_tokens_max, False, True)
    return [tokens[i] for i in range(n)] if n >= 0 else []

def token_to_bytes(vocab, token_id):
    buf = ctypes.create_string_buffer(256)
    n = llama_token_to_piece(vocab, token_id, buf, ctypes.sizeof(buf), 0, True)
    return buf.raw[:n] if n > 0 else b""

def get_token_embeddings_gguf(model_path):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    cache_path = os.path.join(os.path.dirname(model_path), f"{model_name}.embd.npy")
    
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(model_path):
        return np.load(cache_path)
    
    reader = gguf.GGUFReader(model_path, mode='r')
    
    # 获取 Embedding 维度 (Hidden Size)
    n_embd = 1024 # 默认值
    if "llama.embedding_length" in reader.metadata:
        n_embd = int(reader.metadata["llama.embedding_length"][0])
    
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            if t.tensor_type == 8: # Q8_0
                data_u8 = np.frombuffer(t.data, dtype=np.uint8)
                n_blocks = data_u8.size // 34
                blocks = data_u8.reshape(n_blocks, 34)
                deltas = blocks[:, :2].view(np.float16).flatten()
                quants = blocks[:, 2:].view(np.int8)
                data = (deltas[:, np.newaxis] * quants).flatten().astype(np.float32).reshape(-1, n_embd)
            else:
                data = t.data.astype(np.float32) if t.data.dtype == np.float16 else t.data
            np.save(cache_path, data)
            return data
    return None
