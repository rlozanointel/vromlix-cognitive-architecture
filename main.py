# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=1.68.0", "instructor>=1.7.0", "tenacity>=9.0.0",
#     "httpx>=0.28.1", "numpy>=2.2.6", "pydantic>=2.12.5", "sqlite-vec>=0.1.9",
#     "duckduckgo-search>=8.1.1", "feedparser>=6.0.12", "lxml>=5.1.0",
#     "tqdm>=4.67.3", "markitdown>=0.0.1a4"
# ]
# ///

#!/usr/bin/env -S uv run
# -*- coding: utf-8 -*-
"""
Vromlix Prime V2.0: Infrastructure, Short-Term Memory and Token Monitor.
"""

import concurrent.futures
import functools
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from vromlix_utils import OSINTGrounder, vromlix

try:
    import sqlite_vec

    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False
    logging.warning("⚠️ sqlite-vec not installed. Deep memory (RAG) will be disabled.")


@runtime_checkable
class VromlixBackend(Protocol):
    """Structural contract to decouple core_vromlix_prime from vromlix_utils."""

    @property
    def paths(self) -> Any: ...
    @property
    def config(self) -> Any: ...
    def get_model(self, role: str) -> str: ...
    def get_api_key(self, provider: str = "gemini") -> str | None: ...
    def get_secret(self, key_name: str) -> Any: ...
    def get_safety_settings(self) -> list[Any]: ...
    def get_model_capabilities(self, role: str) -> dict | None: ...
    def query_universal_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        role: str = "VOLUMEN",
        response_model: Any = None,
        tools: list | None = None,
        thinking: bool = False,
    ) -> Any: ...
    def query_local_ollm(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str: ...
    def get_embeddings(self, text: str, role: str = "EMBEDDINGS") -> list[float]: ...
    def report_exhaustion(
        self, key: str, provider: str = "gemini", error_msg: str = ""
    ) -> None: ...


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
def _cached_read_file(filepath_str: str, _mtime: float) -> str:
    """Module-level cached file reader, keyed by (path, mtime). Avoids B019."""
    try:
        with Path(filepath_str).open(encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading {filepath_str}: {e}")
        return ""


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


class ExecutionStep(BaseModel):
    step_id: str = Field(description="Unique ID for this step, e.g., 'step_1'")
    expert_id: str = Field(description="ID of the selected expert")
    required_files: list[str] = Field(
        default_factory=list, description="Exact filenames required by this expert"
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of step_ids that must finish before this step starts",
    )


class SimulatedPath(BaseModel):
    path_logic: str = Field(description="Description of this potential routing path")
    success_probability: float = Field(description="Estimated probability of success (0.0 to 1.0)")


class RoutingResult(BaseModel):
    mcts_simulations: list[SimulatedPath] = Field(
        description="Simulate at least 2 different routing paths before deciding"
    )
    internal_analysis: str = Field(
        description="Logic on why the winning DAG structure was chosen over the alternatives"
    )
    execution_plan: list[ExecutionStep] = Field(
        description="List of execution steps forming the winning Directed Acyclic Graph"
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Array of search queries, empty if none needed",
    )


class MoERouter:
    """Advanced Semantic Router (Mixture of Experts)."""

    def __init__(
        self,
        moe_json_content: str,
        monitor: TokenMonitor,
        router_prompt: str,
        backend: VromlixBackend,
    ):
        self._backend = backend
        self.model_id = backend.get_model("VOLUMEN")
        embed_cfg = backend.get_secret("EMBEDDINGS")
        self.embed_model = embed_cfg["model_id"] if embed_cfg else "gemini-embedding-2-preview"
        self.monitor = monitor
        self.router_prompt = router_prompt
        self.cache_db_path = backend.paths.databases / "cache.sqlite"
        self._init_cache_db()

        try:
            self.moe_data = json.loads(moe_json_content)
        except json.JSONDecodeError:
            logging.error("CRITICAL: Failed to parse MoE Routing data")
            self.moe_data = []
        self.expert_vectors = self._load_expert_vectors()

    def _init_cache_db(self):
        import sqlite3

        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS moe_cache (expert_id TEXT PRIMARY KEY, vector TEXT)"
            )
            conn.commit()

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        import math

        mag1, mag2 = (
            math.sqrt(sum(a * a for a in v1)),
            math.sqrt(sum(b * b for b in v2)),
        )
        return (
            sum(a * b for a, b in zip(v1, v2, strict=False)) / (mag1 * mag2)
            if mag1 and mag2
            else 0.0
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=False,
    )
    def _load_expert_vectors(self) -> dict[str, list[float]]:
        import sqlite3

        vectors = {}

        # Intentar cargar desde SQLite
        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.cursor()
            rows = cursor.execute("SELECT expert_id, vector FROM moe_cache").fetchall()
            if len(rows) == len(self.moe_data) and len(self.moe_data) > 0:
                for row in rows:
                    vectors[row[0]] = json.loads(row[1])
                return vectors

        logging.info("🧠 [Semantic Router] Generating expert embeddings (Cache Miss)...")

        # Generar nuevos vectores
        for exp in self.moe_data:
            text_to_embed = (
                f"Role: {exp['expert_id']}. "
                f"Mechanics: {', '.join(exp.get('mechanics', []))}. "
                f"Instructions: {', '.join(exp.get('instructions', []))}."
            )
            vector = self._backend.get_embeddings(text_to_embed)
            vectors[exp["expert_id"]] = vector

        # Guardar en SQLite
        with sqlite3.connect(self.cache_db_path) as conn:
            for exp_id, vec in vectors.items():
                conn.execute(
                    "INSERT OR REPLACE INTO moe_cache (expert_id, vector) VALUES (?, ?)",
                    (exp_id, json.dumps(vec)),
                )
            conn.commit()

        return vectors

    def _get_expert_profile(self, expert_id: str) -> dict:
        profile = next((e for e in self.moe_data if e["expert_id"] == expert_id), None)
        if not profile:
            profile = next(
                (e for e in self.moe_data if e["expert_id"] == "ORCHESTRATE_SYSTEM_CORE"),
                {},
            )
        return dict(profile) if profile else {}

    def determine_routing(self, user_query: str, recent_context: str) -> dict:
        if not self.moe_data:
            return {
                "execution_plan": [
                    {
                        "step_id": "step_1",
                        "expert_id": "ORCHESTRATE_SYSTEM_CORE",
                        "instructions": ["Fallback mode."],
                        "dynamic_files": [],
                        "depends_on": [],
                    }
                ],
                "search_queries": [],
            }

        if self.expert_vectors and len(recent_context) < 500:
            try:
                query_vector = self._backend.get_embeddings(user_query)
                best_expert, max_sim = None, -1.0
                for exp_id, exp_vector in self.expert_vectors.items():
                    sim = self._cosine_similarity(query_vector, exp_vector)
                    if sim > max_sim:
                        max_sim, best_expert = sim, exp_id

                if max_sim > 0.75 and best_expert:
                    logging.info(
                        f"⚡ [Semantic Router] Short circuit: {best_expert} (Sim: {max_sim:.2f})"
                    )
                    step_dict = self._get_expert_profile(best_expert)
                    step_dict.update({"step_id": "step_1", "dynamic_files": [], "depends_on": []})
                    return {"execution_plan": [step_dict], "search_queries": []}
            except Exception as e:
                logging.warning(f"⚠️ Failure in Semantic Gateway, using LLM Fallback: {e}")

        routing_map = [
            {"id": exp["expert_id"], "cluster": exp["parent_cluster"]} for exp in self.moe_data
        ]
        prompt = self.router_prompt.format(
            recent_context=recent_context,
            user_query=user_query,
            routing_map=json.dumps(routing_map, indent=2),
            current_date=datetime.now().strftime("%Y-%m-%d"),
            current_year=datetime.now().strftime("%Y"),
        )

        try:
            result = self._backend.query_universal_llm(
                system_prompt="You are the MoE Router. Extract structured routing information.",
                user_prompt=prompt,
                role="VOLUMEN",
                response_model=RoutingResult,
            )
            logging.info(f"🧠 [Router CoT]: {result.internal_analysis}")

            execution_plan = []
            for step in result.execution_plan[:3]:
                step_dict = self._get_expert_profile(step.expert_id)
                step_dict.update(
                    {
                        "step_id": step.step_id,
                        "dynamic_files": step.required_files,
                        "depends_on": step.depends_on,
                    }
                )
                execution_plan.append(step_dict)

            dag_steps = [s["step_id"] + "(" + s["expert_id"] + ")" for s in execution_plan]
            logging.info(f"🔀 MoE Router: DAG Planned -> {dag_steps}")
            if result.search_queries:
                logging.info(f"🌐 MoE Router: Web Search requested -> {result.search_queries}")
            return {
                "execution_plan": execution_plan,
                "search_queries": result.search_queries,
            }
        except Exception as e:
            logging.warning(f"Failure in MoE Routing ({e}). Using Fallback.")
            fallback_profile = self._get_expert_profile("ORCHESTRATE_SYSTEM_CORE")
            fallback_profile.update({"step_id": "step_1", "dynamic_files": [], "depends_on": []})
            return {"execution_plan": [fallback_profile], "search_queries": []}


def leer_lineas_de_archivo(filepath: str, linea_inicio: int = 1, linea_fin: int = 100) -> str:
    """Read a specific range of lines from a local file to avoid memory saturation."""
    try:
        target_path = vromlix.paths.base / filepath
        if not target_path.exists() or not target_path.is_file():
            target_path = Path(vromlix.paths.config_xml) / filepath
            if not target_path.exists():
                return f"[ERROR TOOL]: The file {filepath} does not exist."

        with target_path.open(encoding="utf-8") as f:
            lineas = f.readlines()
        total_lineas = len(lineas)
        inicio_idx, fin_idx = max(0, linea_inicio - 1), min(total_lineas, linea_fin)

        if inicio_idx >= total_lineas:
            return (
                f"[ERROR TOOL]: linea_inicio ({linea_inicio}) is greater than"
                f" total lines ({total_lineas})."
            )

        # Satisfy IDE linter indexing
        slice_lines = [lineas[k] for k in range(inicio_idx, fin_idx)]
        header = f"--- FILE: {filepath} (Lines {linea_inicio} to {fin_idx} of {total_lineas}) ---"
        return f"{header}\n{''.join(slice_lines)}"
    except Exception as e:
        return f"[ERROR TOOL]: Failed to read {filepath} -> {e!s}"


class AgenticExecutor:
    """Multi-Agent Execution Engine. Executes multiple experts in parallel (Threading)."""

    def __init__(
        self,
        master_prompt: str,
        tracker: SessionTracker,
        monitor: TokenMonitor,
        repo_file: Path,
        backend: VromlixBackend,
    ):
        self._backend = backend
        self.master_prompt = master_prompt
        self.tracker = tracker
        self.monitor = monitor
        self.repo_file = repo_file

    def _execute_single_expert(
        self,
        expert_profile: dict,
        user_query: str,
        recent_context: str,
        web_context: str,
        retrieved_rag: str,
    ) -> dict:
        expert_id = expert_profile.get("expert_id", "UNKNOWN")
        dynamic_files = expert_profile.get("dynamic_files", [])
        if dynamic_files:
            logging.info(f"📂 Injecting to [{expert_id}]: {dynamic_files}")

        expert_context = f"""
        === ACTIVE EXPERT OVERRIDE ===
        You are currently operating as: {expert_id}
        Mechanics: {", ".join(expert_profile.get("mechanics", []))}
        Constraints: {", ".join(expert_profile.get("constraints", []))}
        Instructions: {", ".join(expert_profile.get("instructions", []))}
        Output Signature: {expert_profile.get("output_signature", "")}
        """

        if web_context:
            expert_context += (
                f"\nCRITICAL DIRECTIVE - TEMPORAL ANCHOR & ANTI-HALLUCINATION GUARDRAIL:"
                f"\n1. The absolute current date is {datetime.now().strftime('%Y-%m-%d')}."
                "\n2. LIVE WEB GROUNDING CONTEXT provided. You MUST prioritize it."
                "\n3. Discard outdated internal knowledge if newer web context exists."
                "\n4. Cross-reference web results with the provided RAG memory."
                "\n5. DO NOT extrapolate, guess, or invent unconfirmed version numbers."
            )

        final_system_instruction = self.master_prompt + "\n" + expert_context
        full_user_prompt = f"{recent_context}\n\n"

        if web_context:
            full_user_prompt += (
                f"=== LIVE WEB GROUNDING CONTEXT (DEEP RESEARCH) ===\n"
                f"{web_context}\n==================================================\n\n"
            )
        if retrieved_rag:
            full_user_prompt += (
                f"=== DEEP MEMORY CONTEXT ===\n{retrieved_rag}\n===========================\n\n"
            )

        web_len = len(web_context) if web_context else 0
        rag_len = len(retrieved_rag) if retrieved_rag else 0
        logging.info(
            f"🔍 [X-Ray - {expert_id}] Master: {len(final_system_instruction)}"
            f" | Web: {web_len} | RAG: {rag_len}"
        )

        if dynamic_files:
            full_user_prompt += (
                "=== RELEVANT FILES IDENTIFIED ===\n"
                "The system detected the following files."
                " Use 'leer_lineas_de_archivo' to inspect them.\n"
            )
            for fname in dynamic_files:
                full_user_prompt += f"- {fname}\n"
            full_user_prompt += "=========================================\n\n"

        full_user_prompt += f"USER QUERY:\n{user_query}"

        if "LOCAL_" in expert_id:
            logging.info(f"🔋 [Local Engine] Requesting inference from Ollama for {expert_id}...")
            respuesta_local = self._backend.query_local_ollm(
                model_name=self._backend.get_model("VOLUMEN"),
                system_prompt=final_system_instruction,
                user_prompt=full_user_prompt,
            )
            return {"expert_id": expert_id, "response": respuesta_local}

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=1, max=4),
            reraise=False,
        )
        def _call_with_reflection(prompt: str) -> tuple[str, str]:
            res = self._backend.query_universal_llm(
                system_prompt=final_system_instruction,
                user_prompt=prompt,
                role="PRECISION",
                tools=[leer_lineas_de_archivo],
                thinking=True,
            )
            text = res.text if hasattr(res, "text") else str(res)
            thoughts = (
                f"\n[🧠 REASONING]:\n{res.thoughts}\n" if getattr(res, "thoughts", None) else ""
            )
            if hasattr(res, "usage_metadata") and res.usage_metadata:
                self.monitor.add_usage(expert_id, res.usage_metadata)

            sig = expert_profile.get("output_signature", "")
            marker = next((m for m in ["```python", "```bash"] if m in sig), None)
            if not marker and "###" in sig:
                marker = sig.split("\n")[0].strip()
            if marker and marker not in text:
                raise ValueError(f"Signature '{marker}' missing — triggering retry")
            return text, thoughts

        try:
            text, thoughts = _call_with_reflection(full_user_prompt)
            return {
                "expert_id": expert_id,
                "response": (thoughts + "\n" + text).strip() if thoughts else text.strip(),
            }
        except Exception as e:
            logging.error(f"❌ Critical error in expert {expert_id}: {e}")
            return {"expert_id": expert_id, "response": f"INTERNAL ERROR: {e}"}

    def process_swarm(
        self,
        user_query: str,
        routing_data: dict,
        recent_context: str,
        retrieved_rag: str = "",
        web_context: str = "",
    ) -> dict[str, str]:
        self.tracker.log_interaction("User", user_query)
        execution_plan = routing_data.get("execution_plan", [])
        if not execution_plan:
            return {}

        logging.info(f"🧠 Executing DAG Engine ({len(execution_plan)} steps in queue)...")
        pending_steps = {step["step_id"]: step for step in execution_plan}
        completed_responses: dict[str, str] = {}
        in_progress: dict[concurrent.futures.Future, str] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(execution_plan)) as executor:
            while pending_steps or in_progress:
                ready_steps = [
                    step_id
                    for step_id, step in pending_steps.items()
                    if all(dep in completed_responses for dep in step.get("depends_on", []))
                ]

                for step_id in ready_steps:
                    step = pending_steps.pop(step_id)
                    inherited_context = "".join(
                        [
                            f"--- INHERITED OUTPUT FROM STEP: {dep} ---\n"
                            f"{completed_responses[dep]}\n\n"
                            for dep in step.get("depends_on", [])
                        ]
                    )
                    augmented_query = (
                        f"=== CONTEXT FROM PREVIOUS EXPERTS (DEPENDENCIES) ===\n"
                        f"{inherited_context}\nORIGINAL USER QUERY:\n{user_query}"
                        if inherited_context
                        else user_query
                    )

                    future = executor.submit(
                        self._execute_single_expert,
                        step,
                        augmented_query,
                        recent_context,
                        web_context,
                        retrieved_rag,
                    )
                    in_progress[future] = step_id
                    time.sleep(1.5)

                if in_progress:
                    done, _ = concurrent.futures.wait(
                        in_progress.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        step_id = in_progress.pop(future)
                        try:
                            result = future.result()
                            completed_responses[step_id] = (
                                f"[{result['expert_id']}]:\n{result['response']}"
                            )
                        except Exception as e:
                            logging.error(f"Error in DAG step {step_id}: {e}")
                            completed_responses[step_id] = (
                                f"[{step.get('expert_id', 'UNKNOWN')} ERROR]: {e}"
                            )

        return completed_responses


class OckhamSynthesizer:
    """Master Synthesizer. Merges multiple expert responses and applies strict Red Team auditing."""

    def __init__(
        self,
        master_prompt: str,
        monitor: TokenMonitor,
        fusion_prompt: str,
        auditor_prompt: str,
        backend: VromlixBackend = vromlix,
    ):
        self.master_prompt = master_prompt
        self.monitor = monitor
        self.fusion_prompt = fusion_prompt
        self.auditor_prompt = auditor_prompt
        self._backend = backend

    def _call_llm(self, prompt: str) -> str:
        return self._backend.query_universal_llm(
            system_prompt="", user_prompt=prompt, role="PRECISION"
        ).text

    def synthesize(
        self, user_query: str, swarm_responses: dict[str, str], routing_data: dict
    ) -> str:
        if len(swarm_responses) > 1:
            logging.info(f"🔬 [Ockham] Merging {len(swarm_responses)} perspectives...")
            raw_inputs = "".join(
                [
                    f"\n--- PERSPECTIVE FROM {exp_id} ---\n{resp}\n"
                    for exp_id, resp in swarm_responses.items()
                ]
            )
            anchor = (
                f"[SYSTEM ANCHOR: Date={datetime.now().strftime('%Y-%m-%d')}. "
                "ANTI-HALLUCINATION: Eliminate unverified model versions, speculative features, "
                "or hallucinated APIs. Only preserve explicitly verifiable facts.]\n\n"
            )
            fusion_prompt = anchor + self.fusion_prompt.format(
                user_query=user_query, raw_inputs=raw_inputs
            )
            draft_response = self._call_llm(fusion_prompt)
        else:
            draft_response = next(iter(swarm_responses.values()))

        if "ERROR" in draft_response:
            return draft_response

        logging.info("   -> [Auditor] Evaluating integrity and enforcing State Tracker...")
        all_constraints = [
            c for exp in routing_data.get("execution_plan", []) for c in exp.get("constraints", [])
        ]
        constraints_str = "\n".join([f"- {c}" for c in set(all_constraints)])
        return self._call_llm(
            self.auditor_prompt.format(
                constraints_str=constraints_str, draft_response=draft_response
            )
        )


class SandboxFirewall:
    """Intercepts Python code and OS commands generated by the LLM. Applies HitL protocol."""

    def __init__(self):
        self.sandbox_dir: Path = vromlix.paths.sandbox
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def _hitl_prompt(self, message: str) -> bool:
        print(
            "\n"
            + "🛡️" * 20
            + "\n 🛑 FIREWALL: INTERVENTION REQUIRED\n"
            + "🛡️" * 20
            + f"\n 🔹 {message}\n"
            + "-" * 40
        )
        while True:
            choice = input("❓ Do you authorize this action? [Y/n]: ").strip().lower()
            if choice in ["y", "yes", ""]:
                return True
            elif choice in ["n", "no"]:
                return False
        return False  # Dummy return for IDE linter

    def execute_if_present(self, llm_response: str) -> str:
        logs = []
        match = re.search(
            r'```(?:json)?\s*(\{.*?"vromlix_os_action".*?\})\s*```',
            llm_response,
            re.DOTALL,
        ) or re.search(r'(\{.*?"vromlix_os_action".*?\})', llm_response, re.DOTALL)

        if match:
            try:
                plan = json.loads(match.group(1)).get("vromlix_os_action", {})
                action, target_str, source_str, content = (
                    str(plan.get("action", "")),
                    str(plan.get("target_path", "")),
                    str(plan.get("source_path", "")),
                    str(plan.get("content", "")),
                )

                if self._hitl_prompt(
                    f"The agent requests to execute an OS action: {action.upper()} on {target_str}"
                ):
                    try:
                        target_path = (self.sandbox_dir / target_str).resolve()
                        if not target_path.is_relative_to(self.sandbox_dir):
                            raise ValueError("Path Traversal detected. Access denied.")
                        source_path = (
                            (self.sandbox_dir / source_str).resolve() if source_str else None
                        )
                        if source_path and not source_path.is_relative_to(self.sandbox_dir):
                            raise ValueError("Path Traversal detected in source. Access denied.")

                        target_path.parent.mkdir(parents=True, exist_ok=True)

                        if action == "create_file":
                            with target_path.open("w", encoding="utf-8") as f:
                                f.write(content)
                            msg = f"File created successfully at {target_path}"
                            print(f"   ✅ {msg}")
                            logs.append(msg)
                        elif action == "delete_file":
                            if target_path.exists():
                                target_path.unlink()
                                msg = f"File deleted successfully: {target_path}"
                                print(f"   ✅ {msg}")
                                logs.append(msg)
                            else:
                                msg = f"File not found for deletion: {target_path}"
                                print(f"   ⚠️ {msg}")
                                logs.append(msg)
                        elif action == "move_file":
                            if source_path and source_path.exists():
                                shutil.move(str(source_path), str(target_path))
                                msg = f"File moved from {source_path.name} to {target_path.name}"
                                print(f"   ✅ {msg}")
                                logs.append(msg)
                            else:
                                msg = "Source file not found for move operation."
                                print(f"   ⚠️ {msg}")
                                logs.append(msg)
                        else:
                            msg = f"Unknown OS action requested: {action}"
                            print(f"   ⚠️ {msg}")
                            logs.append(msg)
                    except Exception as e:
                        err_msg = f"Error executing OS Action: {e}"
                        print(f"   ❌ {err_msg}")
                        logs.append(err_msg)
                else:
                    logs.append("OS Action cancelled by user.")
            except Exception as e:
                logs.append(f"Error parsing OS Action: {e}")

        patch_blocks = re.finditer(
            r"File:\s*([a-zA-Z0-9_.\-/]+).*?<<<< SEARCH\s*\n(.*?)\n====\s*\n(.*?)\n>>>> REPLACE",
            llm_response,
            re.DOTALL | re.IGNORECASE,
        )
        for match in patch_blocks:
            target_str, search_text, replace_text = (
                match.group(1).strip(),
                match.group(2),
                match.group(3),
            )
            target_name = Path(target_str).name
            sandbox_path = self.sandbox_dir / target_name

            source_path = sandbox_path
            if not source_path.exists():
                alt_path = (
                    Path(target_str)
                    if Path(target_str).is_absolute()
                    else (vromlix.paths.base / target_str).resolve()
                )
                if alt_path.exists() and alt_path.is_relative_to(vromlix.paths.base):
                    source_path = alt_path

            if self._hitl_prompt(
                f"The agent proposes a PATCH (Diff) for {target_name}. Apply and save to SANDBOX?"
            ):
                if source_path.exists():
                    with source_path.open(encoding="utf-8") as f:
                        content = f.read()
                    if search_text in content:
                        new_content, patch_successful = (
                            content.replace(search_text, replace_text),
                            True,
                        )
                    elif search_text.strip() in content:
                        new_content, patch_successful = (
                            content.replace(search_text.strip(), replace_text.strip()),
                            True,
                        )
                    else:
                        patch_successful = False

                    if patch_successful:
                        with sandbox_path.open("w", encoding="utf-8") as f:
                            f.write(new_content)
                        msg = f"Patch applied. File saved in SANDBOX/{target_name}"
                        print(f"   ✅ {msg}")
                        logs.append(msg)
                    else:
                        msg = f"Patch failed: SEARCH block not found in {source_path.name}."
                        print(f"   ❌ {msg}")
                        logs.append(msg)
                else:
                    msg = f"Failed to apply patch: Original file {target_name} not found."
                    print(f"   ❌ {msg}")
                    logs.append(msg)
            else:
                logs.append(f"Patch for {target_name} cancelled by user.")

        return " | ".join(logs) if logs else "No OS/Code actions detected."


class DeepMemoryRetriever:
    """Augmented Retrieval Engine (RAG) connected to sqlite-vec."""

    def __init__(self):
        self.db_path = str(vromlix.paths.databases / "vromlix_memory.sqlite")
        embed_config = vromlix.get_secret("EMBEDDINGS")
        self.embedding_model = (
            embed_config["model_id"] if embed_config else "gemini-embedding-2-preview"
        )

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        if not HAS_SQLITE_VEC or not Path(self.db_path).exists():
            return ""
        query_vector = vromlix.get_embeddings(query)
        if not query_vector:
            return ""

        try:
            db = sqlite3.connect(self.db_path)
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            cursor = db.cursor()

            cursor.execute(
                """
                WITH seed_nodes AS (
                    SELECT m.id, m.source_file, m.network_type, m.confidence_score, v.distance
                    FROM vromlix_vectors v
                    JOIN vromlix_metadata m ON v.id = m.id
                    WHERE v.embedding MATCH ? AND k = ?
                )
                SELECT
                    (SELECT group_concat(content, '\n... [ADJACENT NODE] ...\n')
                     FROM (
                         SELECT content FROM vromlix_metadata
                         WHERE source_file = s.source_file
                         AND id BETWEEN s.id - 1 AND s.id + 1
                         ORDER BY id ASC
                     )
                    ) AS expanded_content,
                    s.network_type, s.confidence_score, s.distance
                FROM seed_nodes s ORDER BY s.distance ASC
            """,
                (json.dumps(query_vector), top_k),
            )

            results = cursor.fetchall()
            db.close()

            if not results:
                return ""

            context_blocks = [
                "=== DEEP MEMORY CONTEXT (HINDSIGHT W-B-O-S) ===",
                "INSTRUCTION: [W]=World Facts. [B]=Experiences. [O]=Opinions. [S]=Summaries.",
            ]
            for i, (content, net_type, conf_score, _distance) in enumerate(results):
                net_label = {
                    "W": "WORLD FACT",
                    "B": "BIOGRAPHICAL EXP",
                    "O": f"OPINION (Conf: {conf_score})",
                    "S": "SUMMARY",
                }.get(net_type, "UNKNOWN")
                context_blocks.extend([f"--- Fragment {i + 1} [{net_label}] ---", content.strip()])

            logging.info(f"📚 RAG: {len(results)} fragments retrieved.")
            return "\n".join(context_blocks)
        except Exception as e:
            logging.error(f"Error in RAG: {e}")
            return ""


class RealTimeVectorizer(threading.Thread):
    """Vectorizes interaction in the background and saves it to sqlite-vec."""

    def __init__(self, interaction_text: str, db_path: str, embedding_model: str):
        super().__init__()
        self.interaction_text = interaction_text
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.daemon = True

    def run(self):
        if not HAS_SQLITE_VEC or not Path(self.db_path).exists():
            return
        try:
            vector = vromlix.get_embeddings(self.interaction_text)
            db = sqlite3.connect(self.db_path)
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO vromlix_metadata"
                " (source_file, chunk_type, content, network_type) VALUES (?, ?, ?, ?)",
                ("LIVE_SESSION", "real_time_memory", self.interaction_text, "B"),
            )
            cursor.execute(
                "INSERT INTO vromlix_vectors (id, embedding) VALUES (?, ?)",
                (cursor.lastrowid, json.dumps(vector)),
            )
            db.commit()
            db.close()
        except Exception as e:
            logging.error(f"Vectorizer failed: {e}")


class SubconsciousUpdater(threading.Thread):
    """Analyzes chat looking for new data for 01_Dynamic_Profile.xml."""

    def __init__(self, interaction_text: str, profile_path: Path, profiler_prompt: str):
        super().__init__()
        self.interaction_text = interaction_text
        self.profile_path = profile_path
        self.profiler_prompt = profiler_prompt
        self.daemon = True

    def run(self):
        prompt = self.profiler_prompt.format(
            interaction_text=self.interaction_text,
            timestamp=datetime.now().strftime("%Y-%m-%d"),
        )
        try:
            res = vromlix.query_universal_llm(system_prompt="", user_prompt=prompt, role="VOLUMEN")
            result = res.text.strip()
            if result != "NONE" and "<user_fact" in result:
                history_path = vromlix.paths.prompts / "sys_roger_historial_biografico.xml"
                if history_path.exists():
                    with history_path.open(encoding="utf-8") as f:
                        content = f.read()
                    if "</historical_archive>" in content:
                        new_content = content.replace(
                            "</historical_archive>",
                            f"  {result}\n</historical_archive>",
                        )
                        tmp_path = history_path.with_suffix(".xml.tmp")
                        try:
                            with tmp_path.open("w", encoding="utf-8") as f:
                                f.write(new_content)
                            tmp_path.replace(history_path)
                        except Exception:
                            if tmp_path.exists():
                                tmp_path.unlink()
        except Exception as e:
            logging.error(f"SubconsciousUpdater failed: {e}")


class DocumentForgeAgent:
    """SOTA Operative Agent (The Forge): Generates complete documents in the Sandbox."""

    def __init__(self, forge_prompt: str):
        self.sandbox_dir = vromlix.paths.sandbox
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.forge_prompt = forge_prompt

    def execute_missions(self, missions_json: str):
        try:
            missions = json.loads(missions_json)
        except json.JSONDecodeError as e:
            print(f"   ❌ Error parsing JSON missions: {e}")
            return

        total = len(missions)
        for index, mission in enumerate(missions, 1):
            target, source, instruction = (
                mission.get("target", "unknown.txt"),
                mission.get("source", "NONE"),
                mission.get("instruction", ""),
            )
            source_content = "No source provided."

            if source != "NONE":
                source_path = self._find_file(source)
                if source_path:
                    try:
                        with source_path.open(encoding="utf-8") as f:
                            source_content = f.read()
                    except Exception as e:
                        source_content = f"Error: {e}"

            prompt = self.forge_prompt.format(
                target=target, instruction=instruction, source_content=source_content
            )

            try:
                res = vromlix.query_universal_llm(
                    system_prompt="", user_prompt=prompt, role="VOLUMEN"
                )
                content = res.text.strip()
                if content.startswith("```") and content.endswith("```"):
                    lines = content.split("\n")
                    if len(lines) > 2:
                        content = "\n".join(lines[1:-1])

                with (self.sandbox_dir / target).open("w", encoding="utf-8") as f:
                    f.write(content)
                sys.stdout.write("\033[K")
                print(f"   ⚡ [{index}/{total}] Mission completed -> SANDBOX/{target}")
            except Exception as e:
                sys.stdout.write("\033[K")
                print(f"   ❌ [{index}/{total}] Error forging '{target}': {e}")

    def _find_file(self, filename: str) -> Path | None:
        for root, _, files in os.walk(vromlix.paths.base):
            if "venv" in root or ".git" in root or "databases" in root:
                continue
            if filename in files:
                return Path(root) / filename
        return None


class VromlixTerminalUI:
    """Main application loop (PAIOS)."""

    def __init__(self):
        self.max_file_size = getattr(vromlix.config, "MAX_FILE_SIZE_MB", 5) if vromlix.config else 5
        print("\n" + "=" * 50 + "\n 🧠 INITIALIZING VROMLIX PRIME OS (v2.0)\n" + "=" * 50)

        self.monitor = TokenMonitor()
        self.loader = VromlixContextLoader()
        self.sys_prompts = self.loader.load_system_prompts()
        self.master_prompt = self.loader.build_master_system_prompt()
        self.tracker = SessionTracker()

        moe_content = self.loader._read_file(self.loader.moe_file)
        self.router = MoERouter(
            moe_content,
            self.monitor,
            self.sys_prompts.get("moe_router", ""),
            backend=vromlix,
        )
        self.executor = AgenticExecutor(
            self.master_prompt,
            self.tracker,
            self.monitor,
            self.loader.repo_file,
            backend=vromlix,
        )
        self.retriever = DeepMemoryRetriever()
        self.synthesizer = OckhamSynthesizer(
            self.master_prompt,
            self.monitor,
            self.sys_prompts.get("ockham_fusion", ""),
            self.sys_prompts.get("ockham_auditor", ""),
        )
        self.firewall = SandboxFirewall()
        self.pending_attachments = ""
        print("✅ Systems Nominal. Multi-Agent Swarm Active. Short-Term Memory Online.\n")

    def run_headless(self, task_query: str) -> str:
        """Executes a single task in the background (For Industry Benchmarks)."""
        logging.info("🤖 [Headless Mode] Processing Benchmark task...")
        recent_context = self.tracker.get_recent_context(max_turns=1)
        routing_data = self.router.determine_routing(task_query, recent_context)
        rag_context = self.retriever.retrieve_context(task_query)
        swarm_responses = self.executor.process_swarm(
            task_query, routing_data, recent_context, rag_context, ""
        )
        return self.synthesizer.synthesize(task_query, swarm_responses, routing_data)

    def start(self):
        try:
            while True:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "salir"]:
                    print("\n🛑 Closing Vromlix Prime. Session saved.")
                    break

                if user_input.startswith("/leer "):
                    filepath = user_input.replace("/leer ", "").strip()
                    target_path = (
                        Path(filepath)
                        if Path(filepath).is_absolute()
                        else vromlix.paths.base / filepath
                    )

                    if target_path.exists() and target_path.is_file():
                        if target_path.stat().st_size > self.max_file_size * 1024 * 1024:
                            print(
                                f"❌ File too large (>{self.max_file_size}MB)."
                                " Operation cancelled to protect memory."
                            )
                            continue
                        try:
                            with target_path.open(encoding="utf-8") as f:
                                file_content = f.read()
                            self.pending_attachments += (
                                f"\n=== ATTACHED FILE: {target_path.name} ===\n{file_content}\n"
                            )
                            print(
                                f"📎 File '{target_path.name}' loaded into memory."
                                " (Use /limpiar to remove it)"
                            )
                        except Exception as e:
                            print(f"❌ Error reading file: {e}")
                    else:
                        print(f"❌ File not found: {target_path}")
                    continue

                if user_input.strip().lower() == "/limpiar":
                    self.pending_attachments = ""
                    print("🧹 Attachment tray cleared.")
                    continue

                if user_input.startswith("/evolucionar "):
                    target_expert = user_input.replace("/evolucionar ", "").strip()
                    print(f"🧬 Starting algorithmic evolution for: {target_expert}...")

                    try:
                        with self.loader.moe_file.open(encoding="utf-8") as f:
                            moe_data = json.load(f)
                        expert_idx = next(
                            (
                                i
                                for i, exp in enumerate(moe_data)
                                if exp["expert_id"] == target_expert
                            ),
                            None,
                        )
                        if expert_idx is None:
                            print(f"❌ Expert '{target_expert}' not found.")
                            continue

                        expert_profile = moe_data[expert_idx]
                        recent_logs = self.tracker.get_recent_context(max_turns=5)

                        maker_prompt = (
                            "Optimize the 'instructions' and 'constraints' of this expert based on"
                            " recent friction. Return ONLY a JSON object with the updated arrays.\n"
                            f"PROFILE: {json.dumps(expert_profile)}\nFRICTION: {recent_logs}"
                        )
                        maker_res = vromlix.query_universal_llm(
                            system_prompt=(
                                "You are an expert system optimizer. Return valid JSON only."
                            ),
                            user_prompt=maker_prompt,
                            role="PRECISION",
                        )
                        proposed_update = json.loads(maker_res.text)

                        print("   🛡️ [Checker] Auditing the proposed mutation...")
                        checker_prompt = (
                            "Compare the ORIGINAL profile with the PROPOSED update."
                            " If critical core identity rules are deleted or it hallucinates,"
                            ' return {"approved": false}; if it safely adds value,'
                            ' return {"approved": true}.'
                            f"\nORIGINAL: {json.dumps(expert_profile)}"
                            f"\nPROPOSED: {json.dumps(proposed_update)}"
                        )
                        checker_res = vromlix.query_universal_llm(
                            system_prompt="You are a safety auditor. Return valid JSON only.",
                            user_prompt=checker_prompt,
                            role="PRECISION",
                        )
                        verdict = json.loads(checker_res.text)

                        if verdict.get("approved"):
                            moe_data[expert_idx]["instructions"] = proposed_update.get(
                                "instructions", expert_profile["instructions"]
                            )
                            moe_data[expert_idx]["constraints"] = proposed_update.get(
                                "constraints", expert_profile["constraints"]
                            )
                            with self.loader.moe_file.open("w", encoding="utf-8") as f:
                                json.dump(moe_data, f, indent=2, ensure_ascii=False)
                            print(f"   ✅ [Evolution Approved] Expert '{target_expert}' mutated.")
                            self.router.expert_vectors = self.router._load_expert_vectors()
                        else:
                            print(
                                "   ❌ [Evolution Rejected] Identity-loss risk detected. Aborted."
                            )
                    except Exception as e:
                        print(f"❌ Error in evolution: {e}")
                    continue

                full_query = (
                    f"{self.pending_attachments}\n\nUSER QUERY:\n{user_input}"
                    if self.pending_attachments
                    else user_input
                )
                recent_context = self.tracker.get_recent_context(max_turns=3)
                routing_data = self.router.determine_routing(full_query, recent_context)

                web_context = ""
                search_queries = routing_data.get("search_queries", [])
                if search_queries:
                    web_context = OSINTGrounder.execute_deep_research(
                        search_queries, self.sys_prompts.get("osint_synthesis", "")
                    )

                rag_context = self.retriever.retrieve_context(full_query)
                swarm_responses = self.executor.process_swarm(
                    full_query, routing_data, recent_context, rag_context, web_context
                )

                for exp_id, resp in swarm_responses.items():
                    self.tracker.log_interaction(f"Raw_Expert_Data [{exp_id}]", resp)

                print("🧠 [Vromlix] -> Synthesizing master response...")
                final_response = self.synthesizer.synthesize(
                    full_query, swarm_responses, routing_data
                )
                firewall_status = self.firewall.execute_if_present(final_response)

                self.tracker.log_interaction("Vromlix", final_response)
                if "No OS/Code actions" not in firewall_status:
                    self.tracker.log_interaction("System_Firewall", firewall_status)

                missions_json = None
                mission_match = re.search(
                    r"::: VROMLIX_MISSIONS :::\s*(.*?)\s*::: END_MISSIONS :::",
                    final_response,
                    re.DOTALL,
                )
                if mission_match:
                    missions_json = mission_match.group(1)

                display_response = re.sub(
                    r"::: VROMLIX_STATE_TRACKER :::.*?::: END_TRACKER :::",
                    "\n*[Tracker Saved in Memory]*",
                    final_response,
                    flags=re.DOTALL,
                )
                display_response = re.sub(
                    r"::: VROMLIX_MISSIONS :::.*?::: END_MISSIONS :::",
                    "\n*[Forge Missions Delegated]*",
                    display_response,
                    flags=re.DOTALL,
                )
                display_response = re.sub(
                    r'```(?:json)?\s*\{.*?"vromlix_os_action".*?\}\s*```',
                    "\n*[System Action Executed]*",
                    display_response,
                    flags=re.DOTALL,
                )

                print(f"\n🤖 Vromlix:\n{display_response}")
                print(f"\n📊 {self.monitor.get_summary()}")

                if missions_json:
                    DocumentForgeAgent(self.sys_prompts.get("document_forge", "")).execute_missions(
                        missions_json
                    )

                interaction_log = f"USER: {full_query}\n\nVROMLIX: {final_response}"
                RealTimeVectorizer(
                    interaction_log,
                    self.retriever.db_path,
                    self.retriever.embedding_model,
                ).start()
                SubconsciousUpdater(
                    interaction_log,
                    self.loader.profile_file,
                    self.sys_prompts.get("subconscious_profiler", ""),
                ).start()

        except KeyboardInterrupt:
            print("\n\n🛑 Manual interruption (Ctrl+C). Session safely saved. Goodbye.")
            sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vromlix Prime OS")
    parser.add_argument("--task", type=str, help="Execute in Headless mode for Benchmarks")
    args = parser.parse_args()

    ui = VromlixTerminalUI()
    if args.task:
        print("\n=== FINAL BENCHMARK OUTPUT ===")
        print(ui.run_headless(args.task))
    else:
        ui.start()
