"""
API Data Models for Spec-to-Silicon Platform
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Agent execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class DesignSpec(BaseModel):
    """Design specification input"""
    module_name: str = Field(..., description="Module name (e.g., SPI_MASTER_001)")
    description: str = Field(..., description="Natural language specification")
    data_width: Optional[int] = Field(8, description="Data width in bits")
    clock_freq: Optional[float] = Field(None, description="Target clock frequency (MHz)")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters")

    class Config:
        schema_extra = {
            "example": {
                "module_name": "SPI_MASTER_001",
                "description": "SPI Master controller with configurable clock polarity and phase",
                "data_width": 8,
                "clock_freq": 100.0,
                "parameters": {
                    "fifo_depth": 8,
                    "cpol": 0,
                    "cpha": 0
                }
            }
        }


class AgentResult(BaseModel):
    """Result from a single agent execution"""
    agent_name: str
    status: AgentStatus
    duration_seconds: float
    output_file: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    generator: Optional[str] = None
    fallback_reason: Optional[str] = None
    llm_attempted: Optional[bool] = None
    rag_context_chars: Optional[int] = None


class PipelineProgress(BaseModel):
    """Real-time pipeline progress"""
    run_id: str
    current_stage: str
    total_stages: int
    completed_stages: int
    percent_complete: float
    current_agent: Optional[str] = None
    agent_status: Dict[str, AgentStatus] = Field(default_factory=dict)
    start_time: datetime
    estimated_completion: Optional[datetime] = None


class PipelineResult(BaseModel):
    """Complete pipeline execution result"""
    run_id: str
    spec: DesignSpec
    status: AgentStatus
    duration_seconds: float

    # Agent results
    a1_rtl_generation: Optional[AgentResult] = None
    a5_style_review: Optional[AgentResult] = None
    a4_lint_cdc: Optional[AgentResult] = None
    a3_constraints: Optional[AgentResult] = None
    a6_synthesis_script: Optional[AgentResult] = None
    yosys_synthesis: Optional[AgentResult] = None

    # Generated artifacts
    rtl_file: Optional[str] = None
    sdc_file: Optional[str] = None
    synthesis_script: Optional[str] = None
    synthesis_report: Optional[str] = None

    # Metrics
    total_lines: int = 0
    total_ports: int = 0
    warnings_count: int = 0
    errors_count: int = 0
    synthesis_success: bool = False

    start_time: datetime
    end_time: Optional[datetime] = None
    verification: "VerificationSummary" = Field(default_factory=lambda: VerificationSummary())


class VerificationTestbench(BaseModel):
    tb_id: str
    path: str
    generator: str
    generated_at: datetime
    validation: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerificationSimulation(BaseModel):
    job_id: str
    status: str
    simulator: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    rtl_path: Optional[str] = None
    testbench_path: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    coverage: Dict[str, Any] = Field(default_factory=dict)


class VerificationSummary(BaseModel):
    testbenches: List[VerificationTestbench] = Field(default_factory=list)
    simulations: List[VerificationSimulation] = Field(default_factory=list)


class SimulationJob(BaseModel):
    job_id: str
    status: str
    simulator: str
    start_time: datetime
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)


class CodeMetrics(BaseModel):
    """RTL code quality metrics"""
    lines_total: int
    lines_code: int
    lines_comment: int
    lines_blank: int
    modules_count: int
    ports_count: int
    parameters_count: int
    has_fsm: bool
    has_fifo: bool
    complexity_score: float


class SynthesisMetrics(BaseModel):
    """Synthesis results"""
    cell_count: int
    wire_count: int
    area_estimate: Optional[float] = None
    critical_path_ns: Optional[float] = None
    max_frequency_mhz: Optional[float] = None
    warnings: int = 0
    errors: int = 0
    success: bool = False


class RunSummary(BaseModel):
    """Summary of a pipeline run for dashboard"""
    run_id: str
    timestamp: datetime
    module_name: str
    status: AgentStatus
    duration_seconds: float
    synthesis_success: bool
    lines_generated: int
    quality_score: float  # 0-10


PipelineResult.model_rebuild()
