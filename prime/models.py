"""
prime.models.py — VROMLIX Prime: Shared Protocols and Pydantic Models
Split from core_vromlix_prime.py (God Class refactor).
"""

from typing import Any, Protocol

from pydantic import BaseModel, Field


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
