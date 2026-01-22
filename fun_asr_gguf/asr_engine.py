"""
ASR 推理引擎模块

提供端到端的语音识别功能，支持：
- ONNX Encoder + GGUF LLM Decoder
- CTC 辅助识别和时间戳对齐
- 热词检索和纠错
- 上下文信息注入
- 多语言支持
"""

import os
import ctypes
import wave
import numpy as np
import time
from typing import Optional, List, Tuple

from . import nano_llama
from .nano_ctc import load_ctc_tokens, decode_ctc, align_timestamps
from .nano_audio import load_audio
from .nano_onnx import load_onnx_models, encode_audio
from .hotword.hot_phoneme import PhonemeCorrector
from .types import (
    RecognitionResult,
    RecognitionStream,
    TranscriptionResult,
    DecodeResult,
    Timings,
    ASREngineConfig,
    Statistics,
)


# ==================== Vulkan 选项 ====================

# os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan
# os.environ["GGML_VK_VISIBLE_DEVICES"] = "0"   # 禁止 Vulkan 用独显（强制用集显）
# os.environ["GGML_VK_DISABLE_F16"] = "1"       # 禁止 VulkanFP16 计算（Intel集显fp16有溢出问题）


class FunASREngine:
    """FunASR 推理引擎（兼容 sherpa-onnx API）"""

    def __init__(
        self,
        encoder_onnx_path: str,
        ctc_onnx_path: str,
        decoder_gguf_path: str,
        tokens_path: str,
        hotwords_path: str = None,
        enable_ctc: bool = True,
        n_predict: int = 512,
        n_threads: int = None,
        quiet_mode: bool = True,
    ):
        """
        初始化 ASR 引擎

        Args:
            encoder_onnx_path: Encoder ONNX 模型路径
            ctc_onnx_path: CTC ONNX 模型路径
            decoder_gguf_path: Decoder GGUF 模型路径
            tokens_path: Tokens 文件路径
            hotwords_path: 热词文件路径（可选）
            enable_ctc: 是否启用 CTC 辅助
            n_predict: 最大生成 token 数
            n_threads: 线程数
            quiet_mode: 静默模式
        """
        # 配置参数
        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 模型路径（必须提供）
        self.encoder_onnx_path = encoder_onnx_path
        self.ctc_onnx_path = ctc_onnx_path
        self.decoder_gguf_path = decoder_gguf_path
        self.tokens_path = tokens_path
        self.hotwords_path = hotwords_path or os.path.join(self.script_dir, "hot.txt")

        self.enable_ctc = enable_ctc
        self.hotwords_path = hotwords_path or os.path.join(self.script_dir, "hot.txt")
        self.n_predict = n_predict
        self.n_threads = n_threads or (os.cpu_count() // 2)
        self.n_threads_batch = os.cpu_count()
        self.quiet_mode = quiet_mode

        # 推理参数
        self.sample_rate = 16000
        self.inject_chunk_size = 512
        self.n_ubatch = 512
        self.stop_tokens = [151643, 151645]

        # 模型组件（延迟加载）
        self.encoder_sess = None
        self.ctc_sess = None
        self.model = None
        self.ctx = None  # LLM 上下文（复用）
        self.vocab = None
        self.eos_token = None
        self.embedding_table = None
        self.ctc_id2token = None
        self.corrector = None

        self._initialized = False

    def _log(self, message: str, force: bool = False):
        """打印日志（除非在静默模式）"""
        if not self.quiet_mode or force:
            print(message)

    def initialize(self, verbose: bool = True) -> bool:
        """
        初始化所有模型组件

        Args:
            verbose: 是否打印详细信息

        Returns:
            是否初始化成功
        """
        if self._initialized:
            self._log("[ASR] 模型已初始化", True)
            return True

        try:
            t_start = time.perf_counter()

            # 1. 加载 ONNX 模型
            self._log("[1/6] 加载 ONNX 模型...", verbose)
            self.encoder_sess, self.ctc_sess, _ = load_onnx_models(
                self.encoder_onnx_path,
                self.ctc_onnx_path
            )

            # 2. 加载 GGUF LLM
            self._log("[2/6] 加载 GGUF LLM Decoder...", verbose)
            nano_llama.init_llama_lib()

            model_params = nano_llama.llama_model_default_params()
            self.model = nano_llama.llama_model_load_from_file(
                self.decoder_gguf_path.encode('utf-8'),
                model_params
            )

            if not self.model:
                raise RuntimeError("Failed to load GGUF model")

            self.vocab = nano_llama.llama_model_get_vocab(self.model)
            self.eos_token = nano_llama.llama_vocab_eos(self.vocab)

            # 3. 加载 Embedding 权重
            self._log("[3/6] 加载 Embedding 权重...", verbose)
            self.embedding_table = nano_llama.get_token_embeddings_gguf(self.decoder_gguf_path)
            if self.embedding_table is None:
                raise RuntimeError("Failed to load embedding weights")

            # 4. 创建 LLM 上下文（复用）
            self._log("[4/6] 创建 LLM 上下文...", verbose)
            self.ctx = self._create_context()
            if not self.ctx:
                raise RuntimeError("Failed to create LLM context")

            # 5. 加载 CTC 词表
            self._log("[5/6] 加载 CTC 词表...", verbose)
            self.ctc_id2token = load_ctc_tokens(self.tokens_path)

            # 6. 初始化热词纠错器
            self._log("[6/6] 初始化热词纠错器...", verbose)
            self.corrector = PhonemeCorrector(threshold=0.8)

            if os.path.exists(self.hotwords_path):
                with open(self.hotwords_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    loaded_count = self.corrector.update_hotwords(content)
                    self._log(f"      已加载 {loaded_count} 条热词", verbose)
            else:
                self.corrector.update_hotwords("")

            self._initialized = True

            t_cost = time.perf_counter() - t_start
            self._log(f"✓ 模型加载完成 (耗时: {t_cost:.2f}s)", True)

            return True

        except Exception as e:
            self._log(f"✗ 初始化失败: {e}", True)
            return False

    def _create_context(self):
        """创建 LLM 上下文"""
        ctx_params = nano_llama.llama_context_default_params()
        ctx_params.n_ctx = 2048
        ctx_params.n_batch = 2048
        ctx_params.n_ubatch = self.n_ubatch
        ctx_params.embeddings = False
        ctx_params.no_perf = True
        ctx_params.n_threads = self.n_threads
        ctx_params.n_threads_batch = self.n_threads_batch
        return nano_llama.llama_init_from_model(self.model, ctx_params)

    def _process_audio(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """处理音频文件"""
        audio = load_audio(audio_path, self.sample_rate)
        audio_embd, enc_output = encode_audio(audio, self.encoder_sess)
        return audio_embd, enc_output, len(audio)

    def _run_ctc_decode(self, enc_output: np.ndarray) -> Tuple[List, List[str]]:
        """CTC 解码"""
        if not self.enable_ctc:
            return [], []

        ctc_logits = self.ctc_sess.run(None, {"enc_output": enc_output})[0]
        ctc_text, ctc_results = decode_ctc(ctc_logits, self.ctc_id2token)

        # 热词匹配
        hotwords = []
        if self.corrector and self.corrector.hotwords and ctc_text:
            res = self.corrector.correct(ctc_text, k=10)
            candidates = set()
            for _, hw, _ in res.matchs:
                candidates.add(hw)
            for _, hw, _ in res.similars:
                candidates.add(hw)
            hotwords = list(candidates)

        return ctc_results, hotwords

    def _prepare_prompt(
        self,
        hotwords: List[str] = None,
        language: Optional[str] = None,
        context: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """准备 Prompt Embeddings

        Args:
            hotwords: 热词列表
            language: 目标语言（如 "中文", "英文", "日文"）
            context: 上下文信息
        """
        # 构建 Prompt
        prefix_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"

        if hotwords or context:
            if context:
                prefix_prompt += f"请结合上下文信息，更加准确地完成语音转写任务。\n\n\n"
                prefix_prompt += f"**上下文信息：**{context}\n\n\n"

            if hotwords:
                hotwords_str = ", ".join(hotwords)
                prefix_prompt += f"热词列表：[{hotwords_str}]\n"

        if not language:  # 空字符串或 None 都不指定语言
            prefix_prompt += "语音转写："
        else:
            prefix_prompt += f"语音转写成{language}："

        suffix_prompt = "<|im_end|>\n<|im_start|>assistant\n"

        # 转换为 embeddings
        prefix_tokens = nano_llama.text_to_tokens(self.vocab, prefix_prompt)
        suffix_tokens = nano_llama.text_to_tokens(self.vocab, suffix_prompt)

        prefix_embd = self.embedding_table[prefix_tokens].astype(np.float32)
        suffix_embd = self.embedding_table[suffix_tokens].astype(np.float32)

        return prefix_embd, suffix_embd, len(prefix_tokens), len(suffix_tokens)

    def _run_llm_decode(
        self,
        full_embd: np.ndarray,
        n_input_tokens: int,
        stream_output: bool = False
    ) -> Tuple[str, int, float, float]:
        """运行 LLM 解码

        Args:
            full_embd: 完整的 embedding
            n_input_tokens: 输入 token 数量
            stream_output: 是否流式输出（实时显示生成文本）

        Returns:
            (生成的文本, 生成token数, 注入时间, 生成时间)
        """

        t_inject_start = time.perf_counter()
        # 清空 KV cache（复用上下文），注入 embeddings
        mem = nano_llama.llama_get_memory(self.ctx)
        nano_llama.llama_memory_clear(mem, True)
        batch_embd = nano_llama.llama_batch_init(n_input_tokens, full_embd.shape[1], 1)
        batch_embd.n_tokens = n_input_tokens
        batch_embd.token = ctypes.cast(None, ctypes.POINTER(nano_llama.llama_token))

        if not full_embd.flags['C_CONTIGUOUS']:
            full_embd = np.ascontiguousarray(full_embd)
        ctypes.memmove(batch_embd.embd, full_embd.ctypes.data, full_embd.nbytes)

        for k in range(n_input_tokens):
            batch_embd.pos[k] = k
            batch_embd.n_seq_id[k] = 1
            batch_embd.seq_id[k][0] = 0
            batch_embd.logits[k] = 1 if k == n_input_tokens - 1 else 0

        ret = nano_llama.llama_decode(self.ctx, batch_embd)
        nano_llama.llama_batch_free(batch_embd)

        if ret != 0:
            raise RuntimeError(f"Decode failed (ret={ret})")

        t_inject = time.perf_counter() - t_inject_start

        # 2. 生成文本
        t_gen_start = time.perf_counter()
        vocab_size = nano_llama.llama_vocab_n_tokens(self.vocab)
        batch_text = nano_llama.llama_batch_init(1, 0, 1)
        batch_text.n_tokens = 1

        generated_text = ""
        current_pos = n_input_tokens
        decoder = nano_llama.ByteDecoder()
        tokens_generated = 0

        for _ in range(self.n_predict):
            logits_ptr = nano_llama.llama_get_logits(self.ctx)
            logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
            token_id = int(np.argmax(logits_arr))

            if token_id == self.eos_token or token_id in self.stop_tokens:
                break

            raw_bytes = nano_llama.token_to_bytes(self.vocab, token_id)
            text_piece = decoder.decode(raw_bytes)
            generated_text += text_piece
            tokens_generated += 1

            # 流式输出
            if stream_output:
                print(text_piece, end="", flush=True)

            batch_text.token[0] = token_id
            batch_text.pos[0] = current_pos
            batch_text.n_seq_id[0] = 1
            batch_text.seq_id[0][0] = 0
            batch_text.logits[0] = 1

            if nano_llama.llama_decode(self.ctx, batch_text) != 0:
                break

            current_pos += 1

        remaining = decoder.flush()
        generated_text += remaining
        if stream_output and remaining:
            print(remaining, end="", flush=True)

        nano_llama.llama_batch_free(batch_text)
        t_gen = time.perf_counter() - t_gen_start

        return generated_text, tokens_generated, t_inject, t_gen

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        context: Optional[str] = None,
        verbose: bool = True
    ) -> TranscriptionResult:
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径
            language: 目标语言（如 "中文", "英文", "日文"）
            context: 上下文信息
            verbose: 是否打印详细信息

        Returns:
            TranscriptionResult 对象，包含识别结果和计时信息
        """
        if not self._initialized:
            raise RuntimeError("ASR engine not initialized. Call initialize() first.")

        result = TranscriptionResult()
        timings = result.timings

        try:
            t_start = time.perf_counter()

            self._log(f"\n{'='*70}", verbose)
            self._log(f"处理音频: {os.path.basename(audio_path)}", verbose)
            self._log(f"{'='*70}", verbose)

            # 1. 加载音频
            self._log("\n[1] 加载音频...", verbose)
            audio = load_audio(audio_path, self.sample_rate)
            audio_len = len(audio)
            audio_duration = audio_len / self.sample_rate
            self._log(f"    音频长度: {audio_duration:.2f}s", verbose)

            # 2. 创建流并解码（复用 decode_stream）
            stream = self.create_stream()
            stream.accept_waveform(self.sample_rate, audio)

            # 调用 decode_stream 并获取内部结果
            decode_result = self.decode_stream(stream, language=language, context=context, verbose=verbose, _return_internal=True)

            # 复制计时信息
            timings.encode = decode_result.timings.encode
            timings.ctc = decode_result.timings.ctc
            timings.prepare = decode_result.timings.prepare
            timings.inject = decode_result.timings.inject
            timings.llm_generate = decode_result.timings.llm_generate
            timings.align = decode_result.timings.align

            # 填充结果
            result.text = decode_result.text
            result.segments = decode_result.aligned or []
            result.hotwords = decode_result.hotwords

            if decode_result.ctc_results:
                ctc_text = ''.join([r.text for r in decode_result.ctc_results])
                result.ctc_text = ctc_text

            # 统计信息
            timings.total = time.perf_counter() - t_start

            if verbose:
                # 使用 Statistics dataclass
                stats = Statistics(
                    audio_duration=audio_duration,
                    n_input_tokens=decode_result.audio_embd.shape[0] + decode_result.n_prefix + decode_result.n_suffix,
                    n_prefix_tokens=decode_result.n_prefix,
                    n_audio_tokens=decode_result.audio_embd.shape[0],
                    n_suffix_tokens=decode_result.n_suffix,
                    n_generated_tokens=decode_result.n_gen,
                )
                stats.tps_in = (decode_result.audio_embd.shape[0] + decode_result.n_prefix + decode_result.n_suffix) / timings.inject if timings.inject > 0 else 0
                stats.tps_out = decode_result.n_gen / timings.llm_generate if timings.llm_generate > 0 else 0

                print(f"\n[统计]\n{stats}")

                # 格式化耗时显示（与 asr_e2e.py 一致）
                print(f"\n[转录耗时]")
                print(f"  - 音频编码： {timings.encode*1000:5.0f}ms")
                print(f"  - CTC解码：  {timings.ctc*1000:5.0f}ms")
                print(f"  - LLM读取：  {timings.inject*1000:5.0f}ms")
                print(f"  - LLM生成：  {timings.llm_generate*1000:5.0f}ms")
                if decode_result.aligned:
                    print(f"  - 时间戳对齐:{timings.align*1000:5.0f}ms")
                print(f"  - 总耗时：   {timings.total:5.2f}s")
                print()

            return result

        except Exception as e:
            self._log(f"\n✗ 转录失败: {e}", True)
            raise

    def create_stream(self, hotwords: Optional[str] = None) -> RecognitionStream:
        """
        创建识别流（兼容 sherpa-onnx API）

        Args:
            hotwords: 热词字符串（可选，暂未实现）

        Returns:
            RecognitionStream 对象
        """
        # hotwords 参数保留用于未来扩展，目前兼容 sherpa-onnx API
        _ = hotwords
        return RecognitionStream(sample_rate=self.sample_rate)

    def decode_stream(
        self,
        stream: RecognitionStream,
        language: Optional[str] = None,
        context: Optional[str] = None,
        verbose: bool = False,
        _return_internal: bool = False
    ):
        """
        解码单个音频流（兼容 sherpa-onnx API）

        Args:
            stream: RecognitionStream 对象，需先调用 accept_waveform() 填充音频
            language: 目标语言（如 "中文", "英文", "日文"）
            context: 上下文信息
            verbose: 是否打印详细信息
            _return_internal: 是否返回内部结果（用于 transcribe）

        Returns:
            如果 _return_internal=True，返回 DecodeResult
            否则返回 None（结果存储在 stream.result 中）
        """
        if stream.audio_data is None:
            raise ValueError("Stream has no audio data. Call accept_waveform() first.")

        if not self._initialized:
            raise RuntimeError("ASR engine not initialized. Call initialize() first.")

        try:
            # 初始化 timings（如果需要返回内部结果）
            timings = Timings() if _return_internal else None

            # 1. 音频编码
            t_encode_start = time.perf_counter() if _return_internal else None
            audio = stream.audio_data

            if verbose and _return_internal:
                print("\n[2] 音频编码...")

            audio_embd, enc_output = encode_audio(audio, self.encoder_sess)
            if _return_internal and timings:
                t_encode = time.perf_counter() - t_encode_start
                timings.encode = t_encode
                if verbose:
                    print(f"    耗时: {t_encode*1000:.2f}ms")

            # 2. CTC 解码
            t_ctc_start = time.perf_counter() if _return_internal else None

            if verbose and _return_internal:
                print("\n[3] CTC 解码...")

            ctc_results, hotwords = self._run_ctc_decode(enc_output)
            if _return_internal and timings:
                t_ctc = time.perf_counter() - t_ctc_start
                timings.ctc = t_ctc
                if verbose and ctc_results:
                    ctc_text = ''.join([r.text for r in ctc_results])
                    print(f"    CTC: {ctc_text}")
                    if hotwords:
                        print(f"    热词: {hotwords}")
                if verbose:
                    print(f"    耗时: {t_ctc*1000:.2f}ms")

            # 3. 准备 Prompt
            t_prepare_start = time.perf_counter() if _return_internal else None

            if verbose and _return_internal:
                print("\n[4] 准备 Prompt...")

            prefix_embd, suffix_embd, n_prefix, n_suffix = self._prepare_prompt(hotwords, language, context)
            if _return_internal and timings:
                timings.prepare = time.perf_counter() - t_prepare_start
                if verbose:
                    print(f"    Prefix: {n_prefix} tokens")
                    print(f"    Suffix: {n_suffix} tokens")

            # 4. LLM 解码
            if verbose and _return_internal:
                print("\n[5] LLM 解码...")
                print("=" * 70)

            full_embd = np.concatenate([
                prefix_embd,
                audio_embd.astype(np.float32),
                suffix_embd
            ], axis=0)

            n_input_tokens = full_embd.shape[0]
            text, n_gen, t_inject, t_gen = self._run_llm_decode(
                full_embd, n_input_tokens, stream_output=verbose
            )
            text = text.strip()

            if _return_internal and timings:
                timings.inject = t_inject
                timings.llm_generate = t_gen

            if verbose and _return_internal:
                print("\n" + "=" * 70)

            # 5. 时间戳对齐
            t_align_start = time.perf_counter() if _return_internal else None
            timestamps = []
            tokens = []
            aligned = None

            if verbose and _return_internal:
                print("\n[6] 时间戳对齐")

            if ctc_results:
                aligned = align_timestamps(ctc_results, text)
                if aligned:
                    tokens = [seg['char'] for seg in aligned]
                    timestamps = [seg['start'] for seg in aligned]

                    if verbose and _return_internal:
                        print(f"    对齐耗时: {(time.perf_counter() - t_align_start)*1000:.2f}ms")
                        print(f"    对齐结果 (前10个字符):")
                        for r in aligned[:10]:
                            print(f"      {r['start']:.2f}s: {r['char']}")
                        if len(aligned) > 10:
                            print("      ...")

            if _return_internal and timings:
                timings.align = time.perf_counter() - t_align_start

            # 6. 设置结果到 stream
            stream.set_result(
                text=text,
                timestamps=timestamps,
                tokens=tokens
            )

            # 返回内部结果（供 transcribe 使用）
            if _return_internal:
                return DecodeResult(
                    text=text,
                    ctc_results=ctc_results,
                    aligned=aligned,
                    audio_embd=audio_embd,
                    n_prefix=n_prefix,
                    n_suffix=n_suffix,
                    n_gen=n_gen,
                    timings=timings,
                    hotwords=hotwords
                )

        except Exception as e:
            if verbose:
                self._log(f"✗ 解码失败: {e}", True)
            raise

    def cleanup(self):
        """清理资源"""
        if self.ctx:
            nano_llama.llama_free(self.ctx)
            self.ctx = None
        if self.model:
            nano_llama.llama_model_free(self.model)
            nano_llama.llama_backend_free()
            self._initialized = False
            self._log("[ASR] 资源已释放", True)


def create_asr_engine(
    encoder_onnx_path: str,
    ctc_onnx_path: str,
    decoder_gguf_path: str,
    tokens_path: str,
    hotwords_path: str = None,
    enable_ctc: bool = True,
    verbose: bool = True,
) -> FunASREngine:
    """
    创建并初始化 FunASR 引擎的便捷函数

    Args:
        encoder_onnx_path: Encoder ONNX 模型路径
        ctc_onnx_path: CTC ONNX 模型路径
        decoder_gguf_path: Decoder GGUF 模型路径
        tokens_path: Tokens 文件路径
        hotwords_path: 热词文件路径（可选）
        enable_ctc: 是否启用 CTC
        verbose: 是否打印信息

    Returns:
        初始化好的 FunASREngine 实例
    """
    engine = FunASREngine(
        encoder_onnx_path=encoder_onnx_path,
        ctc_onnx_path=ctc_onnx_path,
        decoder_gguf_path=decoder_gguf_path,
        tokens_path=tokens_path,
        hotwords_path=hotwords_path,
        enable_ctc=enable_ctc,
        quiet_mode=not verbose,
    )

    if not engine.initialize(verbose=verbose):
        raise RuntimeError("Failed to initialize ASR engine")

    return engine
