import os
import json
import pickle

from datetime import datetime 
from pathlib import Path
from typing import Any , Dict , List , Optional
from dataclasses import dataclass , field , asdict
import logging

logger = logging.getLogger(__name__)
BASE_DIR = Path("data")
MEMORY_DIR = BASE_DIR / "memory"
STRATEGY_DIR =BASE_DIR / "strategies"
VIZ_DIR = BASE_DIR / "visualizations"
TRADE_LOG_DIR = BASE_DIR / "trade_logs"


#ensures that the directories exists
for d in [MEMORY_DIR , STRATEGY_DIR , VIZ_DIR , TRADE_LOG_DIR]:
    d.mkdir(parents=True , exist_ok=True)

@dataclass
class AgentMemory:
    agent_id : str
    conversation : List[Dict] = field(default_factory=list)
    decisions : List[Dict] = field(default_factory=list)
    context : Dict[str, Any] = field(default_factory=dict)
    created_at : str = field(
            default_factory=lambda:datetime.now().isoformat())
    updated_at : str = field(
            default_factory=lambda: datetime.now().isoformat())


class MemoryManager:
    """File-system backed persistent memory for agents."""

    def save_agent_memory(self, memory: AgentMemory) -> str:
        memory.updated_at = datetime.now().isoformat()
        path = MEMORY_DIR / f"{memory.agent_id}.json"
        path.write_text(json.dumps(asdict(memory), indent=2))
        logger.info("Memory saved → %s", path)
        return str(path)

    def load_agent_memory(self, agent_id: str) -> AgentMemory:
        path = MEMORY_DIR / f"{agent_id}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return AgentMemory(**data)
        return AgentMemory(agent_id=agent_id)     # fresh memory

    def append_conversation(
        self, agent_id: str, role: str, content: str
    ) -> None:
        mem = self.load_agent_memory(agent_id)
        mem.conversation.append({
            "role"      : role,
            "content"   : content,
            "timestamp" : datetime.now().isoformat(),
        })
        self.save_agent_memory(mem)

    def append_decision(self, agent_id: str, decision: Dict) -> None:
        mem = self.load_agent_memory(agent_id)
        mem.decisions.append({**decision, "timestamp": datetime.now().isoformat()})
        self.save_agent_memory(mem)

    def update_context(self, agent_id: str, key: str, value: Any) -> None:
        mem = self.load_agent_memory(agent_id)
        mem.context[key] = value
        self.save_agent_memory(mem)

    def get_context(self, agent_id: str) -> Dict:
        return self.load_agent_memory(agent_id).context

    # ── strategy persistence ─────────────────────────────────────────────

    def save_strategy(self, name: str, strategy: Dict) -> str:
        path = STRATEGY_DIR / f"{name}.json"
        path.write_text(json.dumps({
            **strategy,
            "saved_at": datetime.now().isoformat(),
        }, indent=2))
        return str(path)

    def load_strategy(self, name: str) -> Optional[Dict]:
        path = STRATEGY_DIR / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def list_strategies(self) -> List[str]:
        return [p.stem for p in STRATEGY_DIR.glob("*.json")]

    # ── trade log ────────────────────────────────────────────────────────

    def log_trade(self, trade: Dict) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        path  = TRADE_LOG_DIR / f"{today}.json"
        logs  = json.loads(path.read_text()) if path.exists() else []
        logs.append({**trade, "logged_at": datetime.now().isoformat()})
        path.write_text(json.dumps(logs, indent=2))

    def get_trade_log(self, date: Optional[str] = None) -> List[Dict]:
        date = date or datetime.now().strftime("%Y-%m-%d")
        path = TRADE_LOG_DIR / f"{date}.json"
        return json.loads(path.read_text()) if path.exists() else []

    # ── visualization ────────────────────────────────────────────────────

    def save_visualization_meta(self, name: str, meta: Dict) -> str:
        path = VIZ_DIR / f"{name}.json"
        path.write_text(json.dumps({
            **meta, "saved_at": datetime.now().isoformat()
        }, indent=2))
        return str(path)

    def list_visualizations(self) -> List[str]:
        return [p.stem for p in VIZ_DIR.glob("*.json")]
