# trading_system/__init__.py

"""
Trading System Package
======================
Multi-agent AI trading system powered by LangChain + Zerodha Kite.

Package Layout
--------------
trading_system/
├── __init__.py          
├── llm_service.py       
├── memory.py            
├── tools.py             
├── sub_agents.py        
├── main_agent.py        
└── dashboard.py         
"""

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Python version check
# ---------------------------------------------------------------------------
if sys.version_info < (3, 10):
    raise RuntimeError(
        f"trading_system requires Python >= 3.10 "
        f"(running {sys.version})"
    )

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
__version__     = "1.0.0"
__author__      = "Trading System"
__description__ = "Multi-agent AI trading system – LangChain + Zerodha Kite"
__all__         = [
    "LLMConfig",
    "LLMService",
    "MemoryManager",
    "AgentMemory",
    "init_kite",
    "get_kite",
    "MarketDataAgent",
    "AnalysisAgent",
    "StrategyAgent",
    "VisualizationAgent",
    "RiskAgent",
    "ExecutionAgent",
    "OrchestratorAgent",
    "build_orchestrator",
    "configure_logging",
]

# ---------------------------------------------------------------------------
# Package logger
# ---------------------------------------------------------------------------
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
_BASE = Path(__file__).parent / "data"

for _dir in [
    _BASE / "memory",
    _BASE / "strategies",
    _BASE / "visualizations",
    _BASE / "trade_logs",
]:
    _dir.mkdir(parents=True, exist_ok=True)

logger.debug("Data directories ready → %s", _BASE)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
_LAZY_MAP = {
    # llm_service
    "LLMConfig"          : ("trading_system.llm_service", "LLMConfig"),
    "LLMService"         : ("trading_system.llm_service", "LLMService"),
    # memory
    "MemoryManager"      : ("trading_system.memory",      "MemoryManager"),
    "AgentMemory"        : ("trading_system.memory",      "AgentMemory"),
    # tools
    "init_kite"          : ("trading_system.tools",       "init_kite"),
    "get_kite"           : ("trading_system.tools",       "get_kite"),
    # sub_agents
    "MarketDataAgent"    : ("trading_system.sub_agents",  "MarketDataAgent"),
    "AnalysisAgent"      : ("trading_system.sub_agents",  "AnalysisAgent"),
    "StrategyAgent"      : ("trading_system.sub_agents",  "StrategyAgent"),
    "VisualizationAgent" : ("trading_system.sub_agents",  "VisualizationAgent"),
    "RiskAgent"          : ("trading_system.sub_agents",  "RiskAgent"),
    "ExecutionAgent"     : ("trading_system.sub_agents",  "ExecutionAgent"),
    # main_agent
    "OrchestratorAgent"  : ("trading_system.main_agent",  "OrchestratorAgent"),
    "build_orchestrator" : ("trading_system.main_agent",  "build_orchestrator"),
}


def __getattr__(name: str):
    if name not in _LAZY_MAP:
        raise AttributeError(
            f"module 'trading_system' has no attribute {name!r}"
        )
    import importlib
    module_path, attr = _LAZY_MAP[name]
    module            = importlib.import_module(module_path)
    value             = getattr(module, attr)
    globals()[name]   = value          # cache for next access
    return value

# ---------------------------------------------------------------------------
# configure_logging  (available immediately, no lazy load needed)
# ---------------------------------------------------------------------------
def configure_logging(
    level    : int  = logging.INFO,
    fmt      : str  = "%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    to_file  : bool = False,
    log_file : str  = "data/trading_system.log",
) -> None:
    """
    Configure logging for every module in the package.

    Parameters
    ----------
    level    : logging level          (default INFO)
    fmt      : log format string
    to_file  : also write to a file   (default False)
    log_file : log file path
    """
    root = logging.getLogger("trading_system")
    root.handlers.clear()
    root.setLevel(level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)

    if to_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)

    root.info(
        "trading_system v%s  |  log level → %s",
        __version__,
        logging.getLevelName(level),
    )
