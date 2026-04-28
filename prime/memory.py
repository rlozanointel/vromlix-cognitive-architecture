"""
prime.memory.py — VROMLIX Prime: Short-Term Memory Layer
TokenMonitor, VromlixContextLoader, SessionTracker.
Split from core_vromlix_prime.py (God Class refactor).
"""

import functools
import logging
import re
import threading
import xml.etree.ElementTree as ET
from hashlib import md5
from pathlib import Path

from vromlix_utils import vromlix


def _cached_read_file(filepath_str: str, _mtime: float) -> str:
    """Module-level cached file reader, keyed by (path, mtime). Avoids B019."""
    try:
        with Path(filepath_str).open(encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading {filepath_str}: {e}")
        return ""


class TokenMonitor:
    """Tracks current session token consumption by expert/API. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self.expert_usage: dict[str, dict[str, int]] = {}

    def add_usage(self, expert_id: str, usage_metadata):
        if not usage_metadata:
            return
        with self._lock:
            if expert_id not in self.expert_usage:
                self.expert_usage[expert_id] = {"in": 0, "out": 0}
            self.expert_usage[expert_id]["in"] += (
                getattr(usage_metadata, "prompt_token_count", 0) or 0
            )
            self.expert_usage[expert_id]["out"] += (
                getattr(usage_metadata, "candidates_token_count", 0) or 0
            )

    def get_summary(self) -> str:
        with self._lock:
            if not self.expert_usage:
                return "🪙 Tokens: 0"
            lines = ["📊 Token Breakdown (Session):"]
            total_in, total_out = 0, 0
            for exp, data in self.expert_usage.items():
                lines.append(f"   ├─ [{exp}]: {data['in']} In | {data['out']} Out")
                total_in += data["in"]
                total_out += data["out"]
            lines.append(f"   └─ TOTAL: {total_in} In | {total_out} Out")
            return "\n".join(lines)


@functools.lru_cache(maxsize=64)
class VromlixContextLoader:
    """Loads and merges immutable configuration files with intelligent caching."""

    def __init__(self):
        self._file_cache: dict[str, tuple[str, float]] = {}
        self._master_prompt_cache: str | None = None
        self._prompt_hash: str = ""
        self.cache_ttl_seconds = 300  # 5 minutes
        self.base_path: Path = vromlix.paths.base
        self.prompts_path: Path = vromlix.paths.prompts
        self.logic_file: Path = self._find_file("system_operating_logic.xml")
        self.profile_file: Path = self._find_file("dynamic_profile.xml")
        self.moe_file: Path = self._find_file("moe_routing.json")
        self.repo_file: Path = self._find_file("Project_Atlas.md")
        self.prompts_file: Path = self._find_file("orchestrator_prompts.xml")

    def _find_file(self, filename: str) -> Path:
        for path in [
            self.base_path,
            self.prompts_path,
            vromlix.paths.config_json,
            vromlix.paths.docs,
        ]:
            if (path / filename).exists():
                return path / filename
        return self.base_path / filename

    def load_system_prompts(self) -> dict:
        prompts: dict[str, str] = {}
        if not self.prompts_file.exists():
            logging.error(f"CRITICAL: {self.prompts_file.name} not found.")
            return prompts
        try:
            tree = ET.parse(self.prompts_file)
            for prompt_elem in tree.getroot().findall("prompt"):
                p_id, p_text = prompt_elem.get("id"), prompt_elem.text
                if p_id and p_text:
                    prompts[p_id] = p_text.strip()
        except Exception as e:
            logging.error(f"Error parsing XML prompts: {e}")
        return prompts

    def _read_file_cached(self, filepath: Path) -> str:
        mtime = filepath.stat().st_mtime if filepath.exists() else 0.0
        return _cached_read_file(str(filepath), mtime)

    def _read_file(self, filepath: Path) -> str:
        if not filepath.exists():
            logging.error(f"CRITICAL: Core file not found -> {filepath.name}")
            return f"<!-- ERROR: {filepath.name} MISSING -->"
        try:
            with filepath.open(encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading {filepath.name}: {e}")
            return ""

    def _calculate_prompt_hash(self) -> str:
        hash_input = "".join(
            [
                str(p.stat().st_mtime)
                for p in [self.logic_file, self.profile_file, self.moe_file]
                if p.exists()
            ]
        )
        return md5(hash_input.encode()).hexdigest()

    def _compress_prompt(self, prompt: str) -> str:
        compressed = re.sub(r"\n\s*\n", "\n", prompt)
        compressed = re.sub(r"\s+", " ", compressed).strip()
        if len(prompt) != len(compressed):
            reduction = (len(prompt) - len(compressed)) / len(prompt) * 100
            logging.info(f"🗜️ Prompt compressed: {reduction:.1f}% size reduction")
        return compressed

    def build_master_system_prompt(self) -> str:
        current_hash = self._calculate_prompt_hash()
        if self._master_prompt_cache is None or self._prompt_hash != current_hash:
            logging.info("🧠 Assembling Master System Prompt (Kernel + Profile + MoE)...")
            master_prompt = f"""
You are VROMLIX PRIME, Polymatic Operating System Orchestrator.
You operate strictly under the architectural definitions provided below.
Your cognitive state is externalized in these documents. Do not hallucinate features.

=== 1. SYSTEM OPERATING LOGIC (KERNEL) ===
{self._read_file_cached(self.logic_file)}

=== 2. DYNAMIC PROFILE (THE SOUL) ===
{self._read_file_cached(self.profile_file)}

=== ORCHESTRATOR DIRECTIVES ===
1. Analyze the user's input and the recent conversation history.
2. Adopt the persona, mechanics, and constraints of the assigned Expert(s).
3. ALWAYS append the VROMLIX_STATE_TRACKER at the end of your response.

=== FILE PATCHING PROTOCOL ===
You are a CONSULTATIVE Senior Architect. DO NOT generate code patches proactively.
1. First, analyze the user's request and provide your findings, analysis, or theoretical solution.
2. End your response by ASKING the user: "Do you want me to generate the code patch to apply these
   changes in [filename]?"
3. ONLY if the user explicitly replies with a "Yes" or gives a direct command to patch, you MUST use
   the following exact format to apply surgical patches. DO NOT rewrite the entire file:
📄 File: [filename.ext]
<<<< SEARCH
[Exact lines to find and replace. Must match the original file perfectly]
====
[New lines to insert]
>>>> REPLACE
"""
            self._master_prompt_cache = self._compress_prompt(master_prompt)
            self._prompt_hash = current_hash
        return self._master_prompt_cache if self._master_prompt_cache is not None else ""


class SessionTracker:
    """Manages current session's short-term memory using SQLite."""

    def __init__(self):
        import importlib

        try:
            chat_mod = importlib.import_module("chat_session_manager")
            self.manager = chat_mod.ChatSessionManager()
        except ImportError:
            # Fallback if the path is not correctly resolved in some environments
            logging.error("Failed to dynamically import chat_session_manager.")
            raise

        self.session_id = None

    def start_session(self, model: str = "default", context: str = "") -> str:
        self.session_id = self.manager.create_session(model, context)
        return self.session_id

    def log_interaction(self, role: str, content: str, tokens: int | None = None) -> None:
        if self.session_id:
            self.manager.add_message(self.session_id, role, content, tokens)

    def get_recent_context(self, max_turns: int = 5) -> str:
        if not self.session_id:
            return ""
        try:
            messages = self.manager.get_session_messages(self.session_id)
            recent_messages = (
                messages[-max_turns * 2 :] if len(messages) > max_turns * 2 else messages
            )
            return "\n\n".join(
                [
                    f"{'👤' if msg['role'] == 'user' else '🤖'} {msg['content']}"
                    for msg in recent_messages
                ]
            )
        except Exception as e:
            logging.error(f"Error getting recent context: {e}")
            return ""

    def end_session(self) -> str:
        if self.session_id:
            self.manager.close_session(self.session_id)
            self.session_id = None
        return ""

    def append_state_tracker(
        self, focus: str, locked: str, stack: str, friction: str, loop: str
    ) -> str:
        tracker = (
            f"::: VROMLIX_STATE_TRACKER :::"
            f"\n[FOCUS]::{focus}\n[LOCKED]::{locked}\n[STACK]::{stack}"
            f"\n[FRICTION]::{friction}\n[LOOP]::{loop}\n::: END_TRACKER :::"
        )
        try:
            if self.session_id:
                self.manager.add_message(
                    self.session_id,
                    "system",
                    tracker.strip(),
                    metadata={"type": "state_tracker"},
                )
        except Exception as e:
            logging.error(f"Failed to write tracker: {e}")
        return tracker.strip()
