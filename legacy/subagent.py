from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class SubAgentType(Enum):
    RISK = "risk_analyzer"
    EXECUTION = "execution_engine"
    MONITORING = "monitoring_agent"
    RESEARCH = "research_agent"

@dataclass
class SubAgentConfig:
    "config for a sub-agent"
    name : str
    type : str 
    model : str
    system_prompt : str
    tools : list
    max_iterations int = 5
    timeout_seconds : int = 10
    temperature : float = 0.3


class SubAgentOrchestrator:

