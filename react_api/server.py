#!/usr/bin/env python3
"""
FastAPI Server for React UI Integration

This is a standalone API server that connects the React TypeScript UI
to the existing Python backend (PipelineOrchestrator) without modifying it.

Architecture:
  React UI (Port 3000) → FastAPI (Port 8000) → PipelineOrchestrator → 6 Agents

Features:
- REST API for pipeline execution
- WebSocket for real-time agent logs
- CORS enabled for React development
- Does NOT modify existing backend
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from uuid import uuid4

# Add parent directory to path to import from existing backend
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Import from existing backend (no modifications needed)
from api.pipeline import PipelineOrchestrator
from api.models import DesignSpec, AgentStatus

# FastAPI App
app = FastAPI(
    title="AI Chip Design API",
    description="REST API for React UI integration with multi-agent chip design system",
    version="1.0.0"
)

# CORS - Allow React dev server
allowed_origins = [
    f"http://localhost:{port}" for port in range(3000, 3011)
] + [
    f"http://127.0.0.1:{port}" for port in range(3000, 3011)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in-memory for now, could be Redis/DB later)
pipeline_runs: Dict[str, dict] = {}
log_subscribers: List[WebSocket] = []

# Initialize orchestrator
workspace = Path.cwd()
orchestrator = PipelineOrchestrator(workspace)


# ============================================================================
# Pydantic Models for API
# ============================================================================

class PipelineRunRequest(BaseModel):
    """Request to run the 6-agent pipeline"""
    module_name: str = Field(..., description="Name of the Verilog module to generate")
    description: str = Field(..., description="Natural language specification of the design")
    data_width: Optional[int] = Field(None, description="Data width in bits")
    clock_freq: Optional[float] = Field(None, description="Target clock frequency in MHz")
    parameters: Optional[Dict] = Field(default_factory=dict, description="Additional parameters")
    intent_type: Optional[str] = Field(None, description="Type of design intent")


class AgentResultResponse(BaseModel):
    """Individual agent result"""
    agent_name: str
    status: str
    output: Optional[str] = None
    errors: List[str] = []
    execution_time: Optional[float] = None


class PipelineRunResponse(BaseModel):
    """Response for pipeline execution"""
    run_id: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    agents: List[AgentResultResponse] = []
    final_rtl: Optional[str] = None
    synthesis_report: Optional[str] = None
    duration_seconds: Optional[float] = None
    rtl_lines: Optional[int] = None
    errors_count: Optional[int] = None
    warnings_count: Optional[int] = None
    synthesis_success: Optional[bool] = None


class LogMessage(BaseModel):
    """Real-time log message"""
    timestamp: str
    agent: str
    level: str  # info, warning, error, success
    message: str
    run_id: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "AI Chip Design API",
        "status": "running",
        "version": "1.0.0",
        "backend": "cda-agent-2C1",
        "ui": "React TypeScript"
    }


@app.get("/api/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "orchestrator": "initialized",
        "workspace": str(workspace),
        "active_runs": len([r for r in pipeline_runs.values() if r["status"] == "running"]),
        "total_runs": len(pipeline_runs),
        "websocket_connections": len(log_subscribers)
    }


@app.post("/api/pipeline/run", response_model=PipelineRunResponse)
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """
    Execute the full 6-agent pipeline

    This runs: A1 (RTL Gen) → A2 (Boilerplate) → A3 (Constraints) →
               A4 (Lint) → A5 (Style) → A6 (EDA Scripts) → Yosys Synthesis
    """
    # Generate run ID
    run_id = str(uuid4())

    # Create DesignSpec from request
    spec = DesignSpec(
        module_name=request.module_name,
        description=request.description,
        data_width=request.data_width,
        clock_freq=request.clock_freq,
        parameters=request.parameters or {}
    )

    run_dir = orchestrator.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize run record
    run_record = {
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "spec": request.dict(),
        "agents": [],
        "final_rtl": None,
        "final_rtl_path": None,
        "synthesis_report": None,
        "synthesis_report_path": None,
        "duration_seconds": 0.0,
        "rtl_lines": 0,
        "errors_count": 0,
        "warnings_count": 0,
        "synthesis_success": None,
        "run_dir": str(run_dir),
        "result_path": str(run_dir / "result.json")
    }
    pipeline_runs[run_id] = run_record

    # Broadcast log
    await broadcast_log(run_id, "system", "info", f"Pipeline started: {request.module_name}")

    # Run pipeline in background
    background_tasks.add_task(execute_pipeline, run_id, spec)

    return PipelineRunResponse(
        run_id=run_id,
        status="running",
        started_at=run_record["started_at"],
        agents=[]
    )


@app.get("/api/pipeline/status/{run_id}", response_model=PipelineRunResponse)
async def get_pipeline_status(run_id: str):
    """Get status of a pipeline run"""
    if run_id not in pipeline_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = pipeline_runs[run_id]

    return PipelineRunResponse(
        run_id=run["run_id"],
        status=run["status"],
        started_at=run["started_at"],
        completed_at=run.get("completed_at"),
        agents=[AgentResultResponse(**a) for a in run["agents"]],
        final_rtl=run.get("final_rtl"),
        synthesis_report=run.get("synthesis_report"),
        duration_seconds=run.get("duration_seconds"),
        rtl_lines=run.get("rtl_lines"),
        errors_count=run.get("errors_count"),
        warnings_count=run.get("warnings_count"),
        synthesis_success=run.get("synthesis_success")
    )


def _summarize_result(result, include_quality: bool = True):
    return {
        "run_id": result.run_id,
        "status": result.status.value if isinstance(result.status, AgentStatus) else str(result.status),
        "module_name": result.spec.module_name,
        "synthesis_success": bool(result.synthesis_success),
        "duration_seconds": float(result.duration_seconds or 0.0),
        "start_time": result.start_time.isoformat() if result.start_time else None,
        "end_time": result.end_time.isoformat() if result.end_time else None,
        "rtl_lines": int(result.total_lines or 0),
        "errors_count": int(result.errors_count or 0),
        "warnings_count": int(result.warnings_count or 0),
    }


def _summarize_inflight(run_id: str, record: dict):
    started_at = record.get("started_at")
    started_dt = datetime.fromisoformat(started_at) if started_at else datetime.now()
    duration = (datetime.now() - started_dt).total_seconds()
    return {
        "run_id": run_id,
        "status": record.get("status", "running"),
        "module_name": record.get("spec", {}).get("module_name", "unknown"),
        "synthesis_success": bool(record.get("synthesis_success")) if record.get("synthesis_success") is not None else False,
        "duration_seconds": record.get("duration_seconds", duration),
        "start_time": record.get("started_at"),
        "end_time": record.get("completed_at"),
        "rtl_lines": record.get("rtl_lines", 0),
        "errors_count": record.get("errors_count", 0),
        "warnings_count": record.get("warnings_count", 0),
    }


@app.get("/api/pipeline/runs")
async def list_runs():
    """List all pipeline runs with summary metrics"""
    summaries: Dict[str, dict] = {}

    # Include completed runs from disk
    if orchestrator.runs_dir.exists():
        for run_dir in sorted(orchestrator.runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue
            result_file = run_dir / "result.json"
            if not result_file.exists():
                continue
            try:
                result = orchestrator._load_pipeline_result(run_dir)  # pylint: disable=protected-access
                summaries[result.run_id] = _summarize_result(result)
            except Exception:
                continue

    # Merge in in-flight runs from memory
    for run_id, record in pipeline_runs.items():
        if record.get("status") == "running" or run_id not in summaries:
            summaries[run_id] = _summarize_inflight(run_id, record)

    runs = sorted(summaries.values(), key=lambda x: x.get("start_time") or "", reverse=True)

    return {
        "runs": runs,
        "total": len(runs)
    }


@app.get("/api/runs")
async def list_runs_alias():
    """Alias for legacy frontend compatibility"""
    return await list_runs()


def _get_run_directory(run_id: str) -> Path:
    run_dir = orchestrator.runs_dir / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run_dir


@app.get("/api/runs/{run_id}")
async def get_run_detail(run_id: str):
    run_dir = _get_run_directory(run_id)

    result_data = None
    result_file = run_dir / "result.json"
    if result_file.exists():
        try:
            result = orchestrator._load_pipeline_result(run_dir)  # pylint: disable=protected-access
            result_data = result.model_dump(mode="json", exclude_none=True)
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(status_code=500, detail=f"Failed to load run result: {exc}") from exc
    else:
        in_memory = pipeline_runs.get(run_id, {}).get("result")
        if in_memory:
            result_data = in_memory

    if result_data is None:
        raise HTTPException(status_code=404, detail=f"Result for run {run_id} not found")

    spec_file = run_dir / "spec.json"
    if spec_file.exists():
        spec_data = json.loads(spec_file.read_text())
    else:
        spec_data = pipeline_runs.get(run_id, {}).get("spec")

    files = sorted([p.name for p in run_dir.iterdir() if p.is_file()])

    return {
        "run_id": run_id,
        "result": result_data,
        "spec": spec_data,
        "files": files,
    }


def _safe_filename(filename: str) -> str:
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return safe_name


@app.get("/api/runs/{run_id}/files/{filename}")
async def get_run_file(run_id: str, filename: str):
    run_dir = _get_run_directory(run_id)
    safe_name = _safe_filename(filename)
    file_path = run_dir / safe_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File {filename} not found for run {run_id}")

    try:
        content = file_path.read_text()
    except UnicodeDecodeError:
        content = file_path.read_bytes().decode("utf-8", errors="ignore")

    stat = file_path.stat()

    return {
        "filename": safe_name,
        "content": content,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
    }


@app.get("/api/runs/{run_id}/download/{filename}")
async def download_run_file(run_id: str, filename: str):
    run_dir = _get_run_directory(run_id)
    safe_name = _safe_filename(filename)
    file_path = run_dir / safe_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File {filename} not found for run {run_id}")

    return FileResponse(file_path, filename=safe_name)


# ============================================================================
# WebSocket for Real-Time Logs
# ============================================================================

@app.websocket("/api/pipeline/logs")
async def websocket_logs(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent logs

    Client connects and receives live updates as agents execute
    """
    await websocket.accept()
    log_subscribers.append(websocket)

    try:
        await websocket.send_json({
            "timestamp": datetime.now().isoformat(),
            "agent": "system",
            "level": "info",
            "message": "WebSocket connected",
            "run_id": "system"
        })

        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        log_subscribers.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in log_subscribers:
            log_subscribers.remove(websocket)


# ============================================================================
# Background Pipeline Execution
# ============================================================================

async def execute_pipeline(run_id: str, spec: DesignSpec):
    """
    Execute pipeline in background thread
    This wraps the existing PipelineOrchestrator.run() method
    """
    run = pipeline_runs[run_id]

    try:
        await broadcast_log(run_id, "A1", "info", "Starting RTL generation...")

        # Execute pipeline using existing orchestrator (runs in thread pool)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            orchestrator.execute_pipeline,
            spec,
            run_id
        )

        # Extract results
        if result.status == AgentStatus.SUCCESS:
            run["status"] = "completed"
            await broadcast_log(run_id, "system", "success", "Pipeline completed successfully!")
        else:
            run["status"] = "failed"
            await broadcast_log(run_id, "system", "error", f"Pipeline failed: {result.errors}")

        run["completed_at"] = datetime.now().isoformat()
        run["duration_seconds"] = float(result.duration_seconds or 0.0)
        run["rtl_lines"] = int(result.total_lines or 0)
        run["errors_count"] = int(result.errors_count or 0)
        run["warnings_count"] = int(result.warnings_count or 0)
        run["synthesis_success"] = bool(result.synthesis_success)

        # Persist serialized result for later detail queries
        run["result"] = result.model_dump(mode="json", exclude_none=True)

        # Capture agent outputs
        agent_results: List[AgentResultResponse] = []
        for agent in [
            result.a1_rtl_generation,
            result.a5_style_review,
            result.a4_lint_cdc,
            result.a3_constraints,
            result.a6_synthesis_script,
            result.yosys_synthesis,
        ]:
            if agent is None:
                continue
            agent_results.append(AgentResultResponse(
                agent_name=agent.agent_name,
                status=agent.status.value if isinstance(agent.status, AgentStatus) else str(agent.status),
                output=agent.output_file,
                errors=agent.errors,
                execution_time=agent.duration_seconds,
            ))
        run["agents"] = [agent.model_dump() for agent in agent_results]

        # Load final artifacts for quick access
        if result.rtl_file:
            rtl_path = Path(result.rtl_file)
            run["final_rtl_path"] = str(rtl_path)
            if rtl_path.exists():
                try:
                    run["final_rtl"] = rtl_path.read_text()
                except UnicodeDecodeError:
                    run["final_rtl"] = None

        if result.synthesis_report:
            report_path = Path(result.synthesis_report)
            run["synthesis_report_path"] = str(report_path)
            if report_path.exists():
                try:
                    run["synthesis_report"] = report_path.read_text()
                except UnicodeDecodeError:
                    run["synthesis_report"] = None

    except Exception as e:
        run["status"] = "failed"
        run["completed_at"] = datetime.now().isoformat()
        run["synthesis_success"] = False
        await broadcast_log(run_id, "system", "error", f"Pipeline error: {str(e)}")


async def broadcast_log(run_id: str, agent: str, level: str, message: str):
    """Broadcast log message to all WebSocket subscribers"""
    log_msg = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "level": level,
        "message": message,
        "run_id": run_id
    }

    # Send to all connected clients
    disconnected = []
    for ws in log_subscribers:
        try:
            await ws.send_json(log_msg)
        except Exception:
            disconnected.append(ws)

    # Remove disconnected clients
    for ws in disconnected:
        if ws in log_subscribers:
            log_subscribers.remove(ws)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("🚀 AI Chip Design FastAPI Server")
    print("=" * 80)
    print(f"Workspace: {workspace}")
    print(f"Backend: cda-agent-2C1 (PipelineOrchestrator)")
    print(f"Server: http://localhost:8000")
    print(f"Docs: http://localhost:8000/docs")
    print(f"WebSocket: ws://localhost:8000/api/pipeline/logs")
    print("=" * 80)

    uvicorn.run(
        "react_api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
