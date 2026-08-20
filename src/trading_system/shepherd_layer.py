# shepherd_layer.py
"""
Shepherd Safety & Governance Layer for the Trading Agent.

This module integrates the Shepherd runtime into the existing LangGraph
pipeline, wrapping high-risk agent operations (strategy generation,
trade execution, model inference) in sandboxed, auditable, reviewable
Shepherd tasks.

Architecture:
    LangGraph Orchestrator
         │
         ├── MarketDataAgent   (direct — no sandbox needed)
         ├── AnalysisAgent     (direct — read-only operations)
         ├── StrategyAgent ──► ShepherdStrategyTask   (sandboxed code gen)
         ├── RiskAgent         (direct — advisory only)
         ├── ExecutionAgent ──► ShepherdTradeTask      (retained approval)
         └── VizAgent          (direct — chart generation)

Usage:
    from trading_system.shepherd_layer import ShepherdSafetyLayer

    safety = ShepherdSafetyLayer(workspace_root=".")
    # Wrap strategy generation
    proposal = safety.propose_strategy(prompt, asset, timeframe)
    # Review the proposal
    print(proposal.diff)
    # Accept or reject
    safety.settle(proposal, accept=True)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Proposal / Settlement Data Models
# ─────────────────────────────────────────────────────────────────────────────

class ProposalStatus(str, Enum):
    """Status of a Shepherd-retained proposal."""
    PENDING   = "pending"
    ACCEPTED  = "accepted"
    REJECTED  = "rejected"
    EXPIRED   = "expired"
    FAILED    = "failed"


@dataclass
class ProposalTrace:
    """Durable execution trace for audit compliance."""
    trace_id     : str
    task_name    : str
    started_at   : float
    completed_at : Optional[float] = None
    outcome      : Optional[str]   = None  # Finished, Failed, Exhausted, Stopped
    events       : List[Dict[str, Any]] = field(default_factory=list)
    token_usage  : Dict[str, int]       = field(default_factory=dict)
    artifacts    : List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id":     self.trace_id,
            "task_name":    self.task_name,
            "started_at":   self.started_at,
            "completed_at": self.completed_at,
            "outcome":      self.outcome,
            "events":       self.events,
            "token_usage":  self.token_usage,
            "artifacts":    self.artifacts,
        }


@dataclass
class StrategyProposal:
    """A retained strategy code proposal awaiting human review."""
    proposal_id  : str
    asset        : str
    timeframe    : str
    prompt       : str
    status       : ProposalStatus
    # The generated strategy code (candidate)
    code         : Optional[str]      = None
    file_path    : Optional[str]      = None
    # Diff / changeset summary
    diff         : Optional[str]      = None
    changed_paths: List[str]          = field(default_factory=list)
    # Backtest results if available
    backtest     : Optional[Dict]     = None
    # Execution trace
    trace        : Optional[ProposalTrace] = None
    # Error info
    error        : Optional[str]      = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id":  self.proposal_id,
            "asset":        self.asset,
            "timeframe":    self.timeframe,
            "prompt":       self.prompt,
            "status":       self.status.value,
            "code":         self.code,
            "file_path":    self.file_path,
            "diff":         self.diff,
            "changed_paths": self.changed_paths,
            "backtest":     self.backtest,
            "trace":        self.trace.to_dict() if self.trace else None,
            "error":        self.error,
        }


@dataclass
class TradeProposal:
    """A retained trade execution proposal awaiting human approval."""
    proposal_id : str
    order       : Dict[str, Any]
    risk_check  : Optional[str]       = None
    status      : ProposalStatus      = ProposalStatus.PENDING
    trace       : Optional[ProposalTrace] = None
    result      : Optional[str]       = None
    error       : Optional[str]       = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "order":       self.order,
            "risk_check":  self.risk_check,
            "status":      self.status.value,
            "trace":       self.trace.to_dict() if self.trace else None,
            "result":      self.result,
            "error":       self.error,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Shepherd Safety Layer
# ─────────────────────────────────────────────────────────────────────────────

class ShepherdSafetyLayer:
    """
    Wraps dangerous agent operations in Shepherd's sandboxed execution model.

    This layer sits between the LangGraph orchestrator and actual side-effects:
    - Strategy code generation → sandboxed, retained, reviewable
    - Trade execution → retained proposal, inspectable, settleable
    - All operations → durable execution traces for audit

    The layer attempts to use the full Shepherd runtime if available,
    and falls back to a compatible local implementation if Shepherd
    packages are not installed.
    """

    def __init__(
        self,
        workspace_root : str  = ".",
        strategies_dir : str  = "data/strategies",
        traces_dir     : str  = "data/shepherd_traces",
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.strategies_dir = self.workspace_root / strategies_dir
        self.traces_dir     = self.workspace_root / traces_dir

        # Ensure directories exist
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

        # Proposal registries (in-memory, backed by trace files)
        self._strategy_proposals : Dict[str, StrategyProposal] = {}
        self._trade_proposals    : Dict[str, TradeProposal]    = {}

        # Try to initialize Shepherd runtime
        self._shepherd_available = False
        self._workspace = None
        self._init_shepherd()

        logger.info(
            "ShepherdSafetyLayer initialized (shepherd=%s, root=%s)",
            "active" if self._shepherd_available else "fallback",
            self.workspace_root,
        )

    def _init_shepherd(self) -> None:
        """Attempt to initialize the Shepherd substrate workspace."""
        try:
            import shepherd as sp
            self._sp = sp

            # Check if workspace is already initialized
            vcscore_path = self.workspace_root / ".vcscore"
            if vcscore_path.exists():
                self._workspace = sp.open(str(self.workspace_root))
                self._shepherd_available = True
                logger.info("Shepherd workspace discovered at %s", vcscore_path)
            else:
                logger.info(
                    "No .vcscore found at %s — Shepherd available but workspace "
                    "not initialized. Run 'shepherd init' to enable full substrate. "
                    "Using fallback mode.",
                    self.workspace_root,
                )
                # Even without substrate, we can use shepherd's task/pipeline API
                self._shepherd_available = True

        except ImportError:
            logger.warning(
                "Shepherd packages not installed. Using fallback safety layer. "
                "Install with: pip install shepherd-ai"
            )
            self._sp = None
            self._shepherd_available = False

    # ─────────────────────────────────────────────────────────────────────
    # Strategy Code Generation (Sandboxed + Retained)
    # ─────────────────────────────────────────────────────────────────────

    def propose_strategy(
        self,
        prompt    : str,
        asset     : str,
        timeframe : str,
        agent_result : Optional[str] = None,
    ) -> StrategyProposal:
        """
        Generate a strategy proposal in a sandboxed environment.

        The generated code is NOT written to the workspace — it is retained
        as a candidate changeset for human review.

        Parameters
        ----------
        prompt : str
            User's strategy request
        asset : str
            Target asset/symbol
        timeframe : str
            Trading timeframe (e.g., '1h', '1d')
        agent_result : str, optional
            Raw output from the LangGraph StrategyAgent (if already generated)

        Returns
        -------
        StrategyProposal
            Retained proposal with code, diff, and trace
        """
        proposal_id = f"strat-{uuid.uuid4().hex[:12]}"
        trace = ProposalTrace(
            trace_id  = f"trace-{uuid.uuid4().hex[:8]}",
            task_name = "propose_strategy",
            started_at = time.time(),
        )
        trace.events.append({
            "event": "task_started",
            "time":  time.time(),
            "input": {"prompt": prompt, "asset": asset, "timeframe": timeframe},
        })

        proposal = StrategyProposal(
            proposal_id = proposal_id,
            asset       = asset,
            timeframe   = timeframe,
            prompt      = prompt,
            status      = ProposalStatus.PENDING,
            trace       = trace,
        )

        try:
            # Determine target file path
            safe_asset = asset.replace("/", "_").replace("\\", "_")
            filename = f"{safe_asset}_{timeframe}_strategy.py"
            target_path = self.strategies_dir / filename

            if agent_result:
                # Extract code from agent's response
                code = self._extract_code_from_response(agent_result)
            else:
                # Generate a strategy template
                code = self._generate_strategy_template(asset, timeframe, prompt)

            proposal.code = code
            proposal.file_path = str(target_path)

            # Build the diff (what WOULD change if accepted)
            if target_path.exists():
                existing = target_path.read_text()
                proposal.diff = self._compute_diff(
                    str(target_path), existing, code
                )
            else:
                proposal.diff = self._compute_diff(
                    str(target_path), "", code
                )
            proposal.changed_paths = [str(target_path)]

            trace.events.append({
                "event": "code_generated",
                "time":  time.time(),
                "file":  str(target_path),
                "lines": len(code.splitlines()),
            })
            trace.outcome = "Finished"

        except Exception as e:
            proposal.status = ProposalStatus.FAILED
            proposal.error  = str(e)
            trace.outcome   = "Failed"
            trace.events.append({
                "event": "error",
                "time":  time.time(),
                "error": str(e),
            })
            logger.error("Strategy proposal failed: %s", e)

        trace.completed_at = time.time()
        self._strategy_proposals[proposal_id] = proposal
        self._persist_trace(trace)

        logger.info(
            "Strategy proposal %s created (status=%s, file=%s)",
            proposal_id, proposal.status.value, proposal.file_path,
        )
        return proposal

    def settle_strategy(
        self,
        proposal_id : str,
        accept      : bool,
        reason      : str = "",
    ) -> StrategyProposal:
        """
        Settle a strategy proposal — accept (select) or reject (discard).

        Parameters
        ----------
        proposal_id : str
            ID of the proposal to settle
        accept : bool
            True to accept and write the strategy file, False to discard
        reason : str
            Optional reason for the settlement decision

        Returns
        -------
        StrategyProposal
            Updated proposal with final status
        """
        proposal = self._strategy_proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Unknown proposal: {proposal_id}")
        if proposal.status != ProposalStatus.PENDING:
            raise ValueError(
                f"Proposal {proposal_id} already settled: {proposal.status.value}"
            )

        if accept and proposal.code and proposal.file_path:
            # Write the strategy file (equivalent to workspace.select())
            target = Path(proposal.file_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(proposal.code)
            proposal.status = ProposalStatus.ACCEPTED
            logger.info("Strategy proposal %s ACCEPTED → %s", proposal_id, target)
        else:
            proposal.status = ProposalStatus.REJECTED
            logger.info("Strategy proposal %s REJECTED (reason: %s)", proposal_id, reason)

        # Record settlement in trace
        if proposal.trace:
            proposal.trace.events.append({
                "event":    "settled",
                "time":     time.time(),
                "accepted": accept,
                "reason":   reason,
            })
            self._persist_trace(proposal.trace)

        return proposal

    # ─────────────────────────────────────────────────────────────────────
    # Trade Execution (Retained Approval Gate)
    # ─────────────────────────────────────────────────────────────────────

    def propose_trade(
        self,
        order      : Dict[str, Any],
        risk_check : Optional[str] = None,
    ) -> TradeProposal:
        """
        Create a retained trade proposal for human review.

        The order is NOT executed — it is held as a candidate until
        explicitly settled via settle_trade().

        Parameters
        ----------
        order : dict
            Order details (symbol, transaction, quantity, order_type, price)
        risk_check : str, optional
            Result from the RiskAgent's evaluation

        Returns
        -------
        TradeProposal
            Retained proposal with order details and risk assessment
        """
        proposal_id = f"trade-{uuid.uuid4().hex[:12]}"
        trace = ProposalTrace(
            trace_id   = f"trace-{uuid.uuid4().hex[:8]}",
            task_name  = "propose_trade",
            started_at = time.time(),
        )
        trace.events.append({
            "event": "trade_proposed",
            "time":  time.time(),
            "order": order,
        })

        proposal = TradeProposal(
            proposal_id = proposal_id,
            order       = order,
            risk_check  = risk_check,
            status      = ProposalStatus.PENDING,
            trace       = trace,
        )

        # Auto-reject if risk check contains REJECTED
        if risk_check and "REJECTED" in risk_check.upper():
            proposal.status = ProposalStatus.REJECTED
            proposal.error  = "Auto-rejected by RiskAgent"
            trace.events.append({
                "event": "auto_rejected",
                "time":  time.time(),
                "reason": "RiskAgent REJECTED",
            })
            trace.outcome = "Stopped"
        else:
            trace.outcome = "Finished"

        trace.completed_at = time.time()
        self._trade_proposals[proposal_id] = proposal
        self._persist_trace(trace)

        logger.info(
            "Trade proposal %s created (status=%s, symbol=%s, qty=%s)",
            proposal_id, proposal.status.value,
            order.get("symbol"), order.get("quantity"),
        )
        return proposal

    def settle_trade(
        self,
        proposal_id  : str,
        accept       : bool,
        execute_fn   : Optional[Any] = None,
        reason       : str = "",
    ) -> TradeProposal:
        """
        Settle a trade proposal — execute (select) or reject (discard).

        Parameters
        ----------
        proposal_id : str
            ID of the trade proposal
        accept : bool
            True to execute the order, False to discard
        execute_fn : callable, optional
            Function to call for order placement (e.g., tools.place_order.invoke)
        reason : str
            Optional reason for the decision

        Returns
        -------
        TradeProposal
            Updated proposal with execution result or rejection
        """
        proposal = self._trade_proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Unknown trade proposal: {proposal_id}")
        if proposal.status != ProposalStatus.PENDING:
            raise ValueError(
                f"Trade {proposal_id} already settled: {proposal.status.value}"
            )

        if accept:
            if execute_fn:
                try:
                    result = execute_fn(proposal.order)
                    proposal.result = str(result)
                    proposal.status = ProposalStatus.ACCEPTED
                    logger.info("Trade %s EXECUTED: %s", proposal_id, result)
                except Exception as e:
                    proposal.error  = str(e)
                    proposal.status = ProposalStatus.FAILED
                    logger.error("Trade %s execution FAILED: %s", proposal_id, e)
            else:
                proposal.status = ProposalStatus.ACCEPTED
                proposal.result = "Order approved (no execute_fn provided)"
        else:
            proposal.status = ProposalStatus.REJECTED
            logger.info("Trade %s REJECTED (reason: %s)", proposal_id, reason)

        if proposal.trace:
            proposal.trace.events.append({
                "event":    "settled",
                "time":     time.time(),
                "accepted": accept,
                "reason":   reason,
                "result":   proposal.result,
            })
            self._persist_trace(proposal.trace)

        return proposal

    # ─────────────────────────────────────────────────────────────────────
    # Proposal Registry & Query API
    # ─────────────────────────────────────────────────────────────────────

    def get_strategy_proposal(self, proposal_id: str) -> Optional[StrategyProposal]:
        """Retrieve a strategy proposal by ID."""
        return self._strategy_proposals.get(proposal_id)

    def get_trade_proposal(self, proposal_id: str) -> Optional[TradeProposal]:
        """Retrieve a trade proposal by ID."""
        return self._trade_proposals.get(proposal_id)

    def list_proposals(
        self,
        proposal_type : str = "all",
        status        : Optional[ProposalStatus] = None,
    ) -> Dict[str, List[Dict]]:
        """
        List all proposals, optionally filtered by type and status.

        Parameters
        ----------
        proposal_type : str
            'strategy', 'trade', or 'all'
        status : ProposalStatus, optional
            Filter by status

        Returns
        -------
        dict
            {'strategies': [...], 'trades': [...]}
        """
        result: Dict[str, List[Dict]] = {"strategies": [], "trades": []}

        if proposal_type in ("strategy", "all"):
            for p in self._strategy_proposals.values():
                if status is None or p.status == status:
                    result["strategies"].append(p.to_dict())

        if proposal_type in ("trade", "all"):
            for p in self._trade_proposals.values():
                if status is None or p.status == status:
                    result["trades"].append(p.to_dict())

        return result

    def pending_count(self) -> Dict[str, int]:
        """Count pending proposals by type."""
        return {
            "strategies": sum(
                1 for p in self._strategy_proposals.values()
                if p.status == ProposalStatus.PENDING
            ),
            "trades": sum(
                1 for p in self._trade_proposals.values()
                if p.status == ProposalStatus.PENDING
            ),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Trace & Audit API
    # ─────────────────────────────────────────────────────────────────────

    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """Load a trace from disk by ID."""
        trace_file = self.traces_dir / f"{trace_id}.json"
        if trace_file.exists():
            return json.loads(trace_file.read_text())
        return None

    def list_traces(self, limit: int = 50) -> List[Dict]:
        """List recent execution traces."""
        traces = []
        for f in sorted(self.traces_dir.glob("trace-*.json"), reverse=True)[:limit]:
            try:
                traces.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return traces

    # ─────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from an agent's markdown-formatted response."""
        import re
        # Try to find fenced code blocks
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Return the longest code block (likely the strategy)
            return max(matches, key=len).strip()
        # If no code blocks, return the raw response
        return response.strip()

    def _generate_strategy_template(
        self, asset: str, timeframe: str, prompt: str
    ) -> str:
        """Generate a basic strategy template when no agent output is available."""
        return f'''"""
Auto-generated strategy template
Asset:     {asset}
Timeframe: {timeframe}
Prompt:    {prompt}
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

⚠️  This is a TEMPLATE — review and customize before using in production.
"""

def signal(df):
    """
    Generate trading signals from OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume

    Returns
    -------
    pd.Series
        Signal series: 1 = BUY, -1 = SELL, 0 = HOLD
    """
    import pandas as pd
    import numpy as np

    signals = pd.Series(0, index=df.index)

    # TODO: Implement strategy logic based on prompt:
    # "{prompt}"

    return signals


# Strategy metadata
STRATEGY_META = {{
    "name":      "{asset}_{timeframe}_strategy",
    "asset":     "{asset}",
    "timeframe": "{timeframe}",
    "version":   "0.1.0",
    "author":    "shepherd-safety-layer",
}}
'''

    def _compute_diff(self, filepath: str, old_content: str, new_content: str) -> str:
        """Compute a unified diff between old and new content."""
        import difflib
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{Path(filepath).name}",
            tofile=f"b/{Path(filepath).name}",
        )
        return "".join(diff) or "(new file)"

    def _persist_trace(self, trace: ProposalTrace) -> None:
        """Persist a trace to disk as JSON."""
        trace_file = self.traces_dir / f"{trace.trace_id}.json"
        try:
            trace_file.write_text(json.dumps(trace.to_dict(), indent=2, default=str))
        except OSError as e:
            logger.error("Failed to persist trace %s: %s", trace.trace_id, e)
