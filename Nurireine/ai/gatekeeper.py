"""
Gatekeeper Module

SLM-based message analysis and filtering.
Determines if the bot should respond and what context is needed.
"""

import logging
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
try:
    from llama_cpp import Llama, LlamaGrammar
except (ImportError, TypeError, Exception) as e:
    # Handle both missing package and the specific typing error seen in logs
    import logging
    logging.getLogger(__name__).warning(f"Could not import llama_cpp: {e}. Local SLM will be unavailable.")
    Llama = None
    LlamaGrammar = None

from .. import config
from ..utils.text import normalize_for_bert
from ..debug_server import broadcast_event

logger = logging.getLogger(__name__)


class Gatekeeper:
    """
    Analyzes user input to determine:
    - Whether a response is needed
    - Whether to retrieve from long-term memory
    - What facts to save to memory
    - Updated conversation summary
    
    Uses a two-stage approach:
    1. BERT classifier for quick response-needed check
    2. SLM for detailed analysis (only if BERT says response needed)
    """
    
    def __init__(
        self, 
        slm_repo: str = None, 
        slm_filename: str = None,
        model_dir: Path = None
    ):
        """
        Initialize the Gatekeeper.
        
        Args:
            slm_repo: HuggingFace repo for SLM model
            slm_filename: SLM model filename
            model_dir: Directory to store models
        """
        self.model_dir = model_dir or config.MODELS_DIR
        
        # Initialize components
        self._classifier: Optional[Any] = None
        self._slm: Optional[Llama] = None
        
        self._init_bert_classifier()
        self._init_slm(slm_repo, slm_filename)
    
    def _init_bert_classifier(self) -> None:
        """Initialize the BERT classifier for quick response-needed checks."""
        model_id = config.slm.bert_model_id
        logger.info(f"Loading BERT classifier from {model_id}...")
        
        try:
            model = ORTModelForSequenceClassification.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._classifier = pipeline(
                "text-classification", 
                model=model, 
                tokenizer=tokenizer
            )
            logger.info("BERT classifier ready.")
        except Exception as e:
            logger.error(f"Failed to load BERT classifier: {e}")
            self._classifier = None
    
    def _init_slm(self, repo_id: str = None, filename: str = None) -> None:
        """Initialize the SLM for detailed analysis."""
        if config.slm.provider == "gemini":
            logger.info("Using Gemini API for Gatekeeper (SLM bypassed).")
            from .llm_service import LLMService
            self.llm_service = LLMService()
            return

        repo_id = repo_id or config.slm.model_repo
        filename = filename or config.slm.model_filename
        
        model_path = self._ensure_model(repo_id, filename)
        
        logger.info(f"Loading SLM Gatekeeper from {model_path}...")
        try:
            self._slm = Llama(
                model_path=str(model_path),
                n_ctx=config.slm.context_size,
                n_gpu_layers=config.slm.gpu_layers,
                verbose=True
            )
            logger.info("SLM Gatekeeper ready.")
        except Exception as e:
            logger.error(f"Failed to load SLM model: {e}")
            raise

    def _ensure_model(self, repo_id: str, filename: str) -> Path:
        """
        Ensure the SLM model file exists locally.
        Downloads from HuggingFace if missing.
        """
        local_path = self.model_dir / filename
        
        if local_path.exists():
            logger.info(f"Found local SLM: {local_path}")
            return local_path
            
        logger.info(f"Downloading SLM from {repo_id}...")
        try:
            download_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False
            )
            return Path(download_path)
        except Exception as e:
            logger.error(f"Failed to download SLM: {e}")
            raise

    async def check_response_needed(
        self, 
        context: str, 
        current_input: str
    ) -> Tuple[bool, float, float]:
        """
        Quickly check if response is needed using BERT (Async wrapper).
        """
        if not self._classifier:
            return True, 0.0, 0.0
        
        start_time = time.perf_counter()
        text = f"{context} [SEP] {current_input}"
        
        try:
            # Run blocking inference in executor
            loop = asyncio.get_running_loop()
            
            def _run_inference():
                return self._classifier(text, truncation=True, max_length=512, top_k=None)
            
            result = await loop.run_in_executor(None, _run_inference)
            
            if isinstance(result, list) and isinstance(result[0], list):
                result = result[0]
            
            top_result = result[0] if isinstance(result, list) else result
            
            label = top_result['label']
            score = top_result['score']
            is_positive = label in ['LABEL_1', 'POSITIVE', 'response_needed']
            threshold = config.slm.bert_threshold
            is_needed = is_positive and score >= threshold
            
        except Exception as e:
            logger.error(f"BERT check failed: {e}")
            return True, 0.0, 0.0
            
        latency = time.perf_counter() - start_time
        return is_needed, score, latency

    async def process_turn(
        self, 
        user_input: str, 
        current_summary: str, 
        recent_messages: List[Dict[str, str]],
        is_explicit: bool = False,
        user_id: str = "unknown",
        user_name: str = "unknown",
        skip_bert: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze user input and determine response strategy (Async).
        """
        broadcast_event("gatekeeper_check", {"stage": "start", "input": user_input})
        
        # Stage 1: Quick BERT check (skip if already done by plan_response)
        if skip_bert or is_explicit:
            is_needed = True
            score = 1.0
            latency = 0.0
        else:
            # Prepare BERT Context
            bert_context_parts = []
            for msg in recent_messages:
                content = msg.get('content', '')
                if msg.get('role') == 'assistant':
                    content = "[BOT_RESPONSE]"
                bert_context_parts.append(content)
            bert_context = " [SEP] ".join(bert_context_parts)
            
            is_needed, score, latency = await self.check_response_needed(bert_context, user_input)
            
            broadcast_event("gatekeeper_check", {
                "stage": "bert_result",
                "needed": bool(is_needed),
                "score": float(score),
                "latency": latency
            })
        
        # Early exit if no response needed
        if not is_needed:
            return self._empty_result()
        
        # Stage 2: Detailed Analysis (SLM or Gemini)
        cleaned_input = self._strip_call_names(user_input)
        
        slm_history_lines = []
        for msg in recent_messages:
            role = msg.get('role', 'user').upper()
            content = msg.get('content', '')
            if role == 'USER' and msg.get('user_name'):
                role = msg['user_name']
            elif role == 'ASSISTANT':
                role = "Nurireine"
            
            slm_history_lines.append(f"{role}: {content}")
            
        slm_history = "\n".join(slm_history_lines)
        
        if config.slm.provider == "gemini":
            return await self._run_gemini_analysis(cleaned_input, current_summary, slm_history, user_id, user_name)
        else:
            if not self._slm:
                logger.warning("Local SLM not loaded, using fallback response.")
                return self._fallback_result()
            return await self._run_slm_analysis(cleaned_input, current_summary, slm_history, user_id, user_name)

    async def _run_gemini_analysis(
        self, 
        user_input: str, 
        current_summary: str, 
        recent_history: str,
        user_id: str = "unknown",
        user_name: str = "unknown"
    ) -> Dict[str, Any]:
        """Run analysis using Gemini API - Phase 1 (fast decision) only."""
        broadcast_event("slm_process", {"stage": "start", "provider": "gemini", "phase": 1})
        
        from google.genai import types
        
        # Phase 1: Fast decision (response/retrieval + search query)
        prompt = config.SLM_DECISION_TEMPLATE.format(
            recent_history=recent_history,
            user_input=user_input
        )
        
        response = await self.llm_service.generate_content_async(
            contents=prompt,
            model=config.slm.api_model_id,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )
        
        if response and response.text:
            try:
                result = json.loads(response.text)
                result = self._sanitize_output(result)
                # Add empty fields for phase 2 (will be filled later if needed)
                result.setdefault("guild_facts", [])
                result.setdefault("user_facts", [])
                result.setdefault("summary_updates", {
                    "topic": None, "mood": None, "new_topic": None, "new_point": None, "stage": None
                })
                broadcast_event("slm_process", {"stage": "end", "phase": 1, "result": result})
                return result
            except json.JSONDecodeError:
                 logger.error("Failed to parse JSON from Gemini Gatekeeper Phase 1.")
                 return self._fallback_result()
        
        logger.warning("Gemini Gatekeeper Phase 1 returned empty or null response.")
        return self._fallback_result()
    
    async def _run_gemini_extraction(
        self,
        user_input: str,
        current_summary: str,
        recent_history: str,
        user_id: str = "unknown",
        user_name: str = "unknown"
    ) -> Dict[str, Any]:
        """Run extraction using Gemini API - Phase 2 (lazy extraction)."""
        broadcast_event("slm_process", {"stage": "start", "provider": "gemini", "phase": 2})
        
        from google.genai import types
        
        # Phase 2: Lazy extraction (facts + summary)
        prompt = config.SLM_EXTRACTION_TEMPLATE.format(
            user_id=user_id,
            user_name=user_name,
            current_summary=current_summary,
            recent_history=recent_history,
            user_input=user_input
        )
        
        response = await self.llm_service.generate_content_async(
            contents=prompt,
            model=config.slm.api_model_id,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )
        
        if response and response.text:
            try:
                result = json.loads(response.text)
                result = self._sanitize_output(result)
                broadcast_event("slm_process", {"stage": "end", "phase": 2, "result": result})
                return result
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from Gemini Gatekeeper Phase 2.")
                return {"guild_facts": [], "user_facts": [], "summary_updates": {}}
        
        logger.warning("Gemini Gatekeeper Phase 2 returned empty or null response.")
        return {"guild_facts": [], "user_facts": [], "summary_updates": {}}

    async def _run_slm_analysis(
        self, 
        user_input: str, 
        current_summary: str, 
        recent_history: str,
        user_id: str = "unknown",
        user_name: str = "unknown"
    ) -> Dict[str, Any]:
        """Run Phase 1 SLM analysis (Local) - Decision only."""
        broadcast_event("slm_process", {"stage": "start", "provider": "local", "phase": 1})
        
        prompt = config.SLM_DECISION_TEMPLATE.format(
            recent_history=recent_history,
            user_input=user_input
        )

        def _blocking_slm_inference():
            MAX_RETRIES = 2
            MAX_CONTINUATIONS = 2
            
            for attempt in range(MAX_RETRIES + 1):
                try:
                    accumulated_text = ""
                    current_prompt = prompt
                    
                    for cont in range(MAX_CONTINUATIONS + 1):
                        output = self._slm(
                            current_prompt,
                            max_tokens=config.slm.max_tokens,
                            stop=config.slm.stop_sequences,
                            grammar=None, 
                            echo=False,
                            temperature=0.1,
                            top_p=0.9,
                            top_k=40
                        )
                        
                        chunk = output['choices'][0]['text'].strip()
                        accumulated_text += chunk
                        
                        # Try to parse the accumulated text
                        result = self._try_parse_json(accumulated_text)
                        if result is not None:
                            result = self._sanitize_output(result)
                            # Add empty fields for phase 2
                            result.setdefault("guild_facts", [])
                            result.setdefault("user_facts", [])
                            result.setdefault("summary_updates", {
                                "topic": None, "mood": None, "new_topic": None, "new_point": None, "stage": None
                            })
                            return result
                        
                        start_idx = accumulated_text.find('{')
                        if start_idx == -1:
                            break
                        
                        # JSON started but incomplete — continue generation
                        partial_json = accumulated_text[start_idx:]
                        current_prompt = prompt + "\n```json\n" + partial_json
                    
                    if attempt == MAX_RETRIES:
                        return self._fallback_result()
                    
                except Exception as e:
                    logger.error(f"SLM processing error (Attempt {attempt+1}): {e}")
                    if attempt == MAX_RETRIES:
                        return self._fallback_result()
            return self._fallback_result()

        # Run blocking inference in executor
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _blocking_slm_inference)
        
        broadcast_event("slm_process", {"stage": "end", "phase": 1, "result": result})
        return result
    
    async def run_extraction(
        self,
        user_input: str,
        current_summary: str,
        recent_messages: List[Dict[str, str]],
        user_id: str = "unknown",
        user_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Run Phase 2 extraction (facts + summary) - Can be called separately.
        This is the lazy extraction that happens after response starts.
        """
        cleaned_input = self._strip_call_names(user_input)
        
        slm_history_lines = []
        for msg in recent_messages:
            role = msg.get('role', 'user').upper()
            content = msg.get('content', '')
            if role == 'USER' and msg.get('user_name'):
                role = msg['user_name']
            elif role == 'ASSISTANT':
                role = "Nurireine"
            
            slm_history_lines.append(f"{role}: {content}")
        
        slm_history = "\n".join(slm_history_lines)
        
        if config.slm.provider == "gemini":
            return await self._run_gemini_extraction(cleaned_input, current_summary, slm_history, user_id, user_name)
        else:
            if not self._slm:
                logger.warning("Local SLM not loaded for extraction, skipping.")
                return {"guild_facts": [], "user_facts": [], "summary_updates": {}}
            return await self._run_slm_extraction(cleaned_input, current_summary, slm_history, user_id, user_name)
    
    async def _run_slm_extraction(
        self,
        user_input: str,
        current_summary: str,
        recent_history: str,
        user_id: str = "unknown",
        user_name: str = "unknown"
    ) -> Dict[str, Any]:
        """Run Phase 2 SLM extraction (Local) - Facts and summary."""
        broadcast_event("slm_process", {"stage": "start", "provider": "local", "phase": 2})
        
        prompt = config.SLM_EXTRACTION_TEMPLATE.format(
            user_id=user_id,
            user_name=user_name,
            current_summary=current_summary,
            recent_history=recent_history,
            user_input=user_input
        )

        def _blocking_slm_inference():
            MAX_RETRIES = 2
            MAX_CONTINUATIONS = 2
            
            for attempt in range(MAX_RETRIES + 1):
                try:
                    accumulated_text = ""
                    current_prompt = prompt
                    
                    for cont in range(MAX_CONTINUATIONS + 1):
                        output = self._slm(
                            current_prompt,
                            max_tokens=config.slm.max_tokens,
                            stop=config.slm.stop_sequences,
                            grammar=None,
                            echo=False,
                            temperature=0.1,
                            top_p=0.9,
                            top_k=40
                        )
                        
                        chunk = output['choices'][0]['text'].strip()
                        accumulated_text += chunk
                        
                        # Try to parse the accumulated text
                        result = self._try_parse_json(accumulated_text)
                        if result is not None:
                            result = self._sanitize_output(result)
                            return result
                        
                        start_idx = accumulated_text.find('{')
                        if start_idx == -1:
                            break
                        
                        # JSON started but incomplete — continue generation
                        partial_json = accumulated_text[start_idx:]
                        current_prompt = prompt + "\n```json\n" + partial_json
                    
                    if attempt == MAX_RETRIES:
                        return {"guild_facts": [], "user_facts": [], "summary_updates": {}}
                    
                except Exception as e:
                    logger.error(f"SLM extraction error (Attempt {attempt+1}): {e}")
                    if attempt == MAX_RETRIES:
                        return {"guild_facts": [], "user_facts": [], "summary_updates": {}}
            return {"guild_facts": [], "user_facts": [], "summary_updates": {}}

        # Run blocking inference in executor
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _blocking_slm_inference)
        
        broadcast_event("slm_process", {"stage": "end", "phase": 2, "result": result})
        return result
    
    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Try to extract and parse JSON from text. Returns None if not parseable."""
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return None
        
        json_str = text[start_idx:end_idx + 1]
        try:
            return json.loads(json_str, strict=False)
        except (json.JSONDecodeError, ValueError):
            return None
    
    @staticmethod
    def _strip_call_names(text: str) -> str:
        """
        Strip bot call names and mentions from the beginning of user input.
        e.g. "누리야 유튜브 영상 주소" -> "유튜브 영상 주소"
        """
        import re
        
        # Remove Discord mentions first (e.g. <@12345> or <@!12345>)
        cleaned = re.sub(r'<@!?\d+>\s*', '', text).strip()
        
        # Sort call names by length descending to match longest first
        # e.g. "누리레느" before "누리" to avoid partial stripping
        call_names = sorted(config.bot.call_names, key=len, reverse=True)
        
        # Strip call names from the beginning (with optional trailing particles)
        for name in call_names:
            # Match: call name + optional Korean particles (야, 아, 에게, 한테, etc.) + space
            pattern = rf'^{re.escape(name)}(?:야|아|에게|한테)?\s*'
            cleaned = re.sub(pattern, '', cleaned, count=1)
        
        return cleaned.strip() if cleaned.strip() else text
    
    @staticmethod
    def _sanitize_output(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process SLM output to replace bot call names with 'Nurireine'.
        Small SLMs often ignore template instructions, so this ensures consistency.
        """
        import re
        
        # Build pattern: match call names (longest first) with optional Korean particles
        call_names = sorted(config.bot.call_names, key=len, reverse=True)
        # Pattern: 누리레느|누리레인|누리야|누리|레느 (+ optional particles like 의, 에, 가, 를, etc.)
        names_pattern = "|".join(re.escape(n) for n in call_names)
        pattern = re.compile(rf'(?:{names_pattern})(?:의|에게|한테|가|를|은|는)?(?=\s|[.,!?]|$)')
        
        def clean_str(s: str) -> str:
            if not isinstance(s, str):
                return s
            return pattern.sub("Nurireine", s)
        
        def clean_recursive(obj):
            if isinstance(obj, str):
                return clean_str(obj)
            elif isinstance(obj, dict):
                return {k: clean_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_recursive(item) for item in obj]
            return obj
        
        return clean_recursive(result)
    
    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Return empty result (no response needed)."""
        return {
            "response_needed": False,
            "retrieval_needed": False,
            "search_query": None,
            "guild_facts": [],
            "user_facts": [],
            "summary_updates": {
                "topic": None,
                "mood": None,
                "new_topic": None,
                "new_point": None,
                "stage": None
            }
        }
    
    @staticmethod
    def _fallback_result() -> Dict[str, Any]:
        """Return fallback result (response needed, minimal analysis)."""
        return {
            "response_needed": True,
            "retrieval_needed": False,
            "search_query": None,
            "guild_facts": [],
            "user_facts": [],
            "summary_updates": {
                "topic": None,
                "mood": None,
                "new_topic": None,
                "new_point": None,
                "stage": None
            }
        }
