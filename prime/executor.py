"""
prime.executor.py — VROMLIX Prime: Agentic Execution Engine
AgenticExecutor, OckhamSynthesizer, SandboxFirewall.
Split from core_vromlix_prime.py (God Class refactor).
"""

import concurrent.futures
import json
import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from prime.memory import SessionTracker, TokenMonitor
from prime.models import VromlixBackend
from prime.router import leer_lineas_de_archivo
from vromlix_utils import vromlix


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
        logging.info(
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
                            logging.info(f"   ✅ {msg}")
                            logs.append(msg)
                        elif action == "delete_file":
                            if target_path.exists():
                                target_path.unlink()
                                msg = f"File deleted successfully: {target_path}"
                                logging.info(f"   ✅ {msg}")
                                logs.append(msg)
                            else:
                                msg = f"File not found for deletion: {target_path}"
                                logging.warning(f"   ⚠️ {msg}")
                                logs.append(msg)
                        elif action == "move_file":
                            if source_path and source_path.exists():
                                shutil.move(str(source_path), str(target_path))
                                msg = f"File moved from {source_path.name} to {target_path.name}"
                                logging.info(f"   ✅ {msg}")
                                logs.append(msg)
                            else:
                                msg = "Source file not found for move operation."
                                logging.warning(f"   ⚠️ {msg}")
                                logs.append(msg)
                        else:
                            msg = f"Unknown OS action requested: {action}"
                            logging.warning(f"   ⚠️ {msg}")
                            logs.append(msg)
                    except Exception as e:
                        err_msg = f"Error executing OS Action: {e}"
                        logging.error(f"   ❌ {err_msg}")
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
                        logging.info(f"   ✅ {msg}")
                        logs.append(msg)
                    else:
                        msg = f"Patch failed: SEARCH block not found in {source_path.name}."
                        logging.error(f"   ❌ {msg}")
                        logs.append(msg)
                else:
                    msg = f"Failed to apply patch: Original file {target_name} not found."
                    logging.error(f"   ❌ {msg}")
                    logs.append(msg)
            else:
                logs.append(f"Patch for {target_name} cancelled by user.")

        return " | ".join(logs) if logs else "No OS/Code actions detected."
