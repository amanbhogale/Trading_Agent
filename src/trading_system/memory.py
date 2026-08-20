import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager

load_dotenv()

# Configure DB connection (loaded from env variables)
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "trading_db"),
    "user": os.getenv("DB_USER", "trading_agent"),
    "password": os.getenv("DB_PASSWORD", "zombie612@"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

# Thread-safe database connection pool
try:
    db_pool = ThreadedConnectionPool(minconn=1, maxconn=10, **DB_CONFIG)
    logger.info("Database connection pool initialized successfully ✅")
except Exception as e:
    logger.error("Failed to initialize database connection pool: %s", e)
    raise

def custom_dumps(obj):
    """Serialize object to JSON string with default=str for Timestamp/datetime support."""
    return json.dumps(obj, default=str)

@dataclass
class AgentMemory:
    agent_id : str
    conversation : List[Dict] = field(default_factory=list)
    decisions : List[Dict] = field(default_factory=list)
    context : Dict[str, Any] = field(default_factory=dict)
    created_at : str = field(default_factory=lambda:datetime.now().isoformat())
    updated_at : str = field(default_factory=lambda: datetime.now().isoformat())

class MemoryManager:
    """PostgreSQL-backed persistent memory for agents."""

    def __init__(self):
        pass

    @contextmanager
    def _get_conn(self):
        conn = db_pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            db_pool.putconn(conn)

    def _ensure_agent(self, conn, agent_id: str):
        with conn.cursor() as cur:
            cur.execute("SELECT agent_id FROM agent_memory WHERE agent_id = %s", (agent_id,))
            if not cur.fetchone():
                now = datetime.now()
                cur.execute(
                    "INSERT INTO agent_memory (agent_id, created_at, updated_at, context) VALUES (%s, %s, %s, %s)",
                    (agent_id, now, now, custom_dumps({}))
                )
                conn.commit()

    def load_agent_memory(self, agent_id: str) -> AgentMemory:
        with self._get_conn() as conn:
            self._ensure_agent(conn, agent_id)
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("SELECT created_at, updated_at, context FROM agent_memory WHERE agent_id = %s", (agent_id,))
                row = cur.fetchone()
                
                cur.execute("SELECT role, content, timestamp FROM conversations WHERE agent_id = %s ORDER BY timestamp ASC", (agent_id,))
                conv_rows = cur.fetchall()
                conversation = [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"].isoformat()} for r in conv_rows]
                
                cur.execute("SELECT data, timestamp FROM decisions WHERE agent_id = %s ORDER BY timestamp ASC", (agent_id,))
                dec_rows = cur.fetchall()
                decisions = []
                for r in dec_rows:
                    d = dict(r["data"])
                    d["timestamp"] = r["timestamp"].isoformat()
                    decisions.append(d)

                return AgentMemory(
                    agent_id=agent_id,
                    conversation=conversation,
                    decisions=decisions,
                    context=row["context"],
                    created_at=row["created_at"].isoformat(),
                    updated_at=row["updated_at"].isoformat()
                )

    def save_agent_memory(self, memory: AgentMemory) -> str:
        with self._get_conn() as conn:
            self._ensure_agent(conn, memory.agent_id)
            with conn.cursor() as cur:
                now = datetime.now()
                cur.execute("UPDATE agent_memory SET updated_at = %s, context = %s WHERE agent_id = %s",
                            (now, custom_dumps(memory.context), memory.agent_id))
            conn.commit()
        return f"PostgreSQL DB: {memory.agent_id}"

    def append_conversation(self, agent_id: str, role: str, content: str) -> None:
        with self._get_conn() as conn:
            self._ensure_agent(conn, agent_id)
            with conn.cursor() as cur:
                now = datetime.now()
                cur.execute("INSERT INTO conversations (agent_id, role, content, timestamp) VALUES (%s, %s, %s, %s)",
                            (agent_id, role, content, now))
                cur.execute("UPDATE agent_memory SET updated_at = %s WHERE agent_id = %s", (now, agent_id))
            conn.commit()

    def append_decision(self, agent_id: str, decision: Dict) -> None:
        with self._get_conn() as conn:
            self._ensure_agent(conn, agent_id)
            with conn.cursor() as cur:
                now = datetime.now()
                cur.execute("INSERT INTO decisions (agent_id, data, timestamp) VALUES (%s, %s, %s)",
                            (agent_id, custom_dumps(decision), now))
                cur.execute("UPDATE agent_memory SET updated_at = %s WHERE agent_id = %s", (now, agent_id))
            conn.commit()

    def update_context(self, agent_id: str, key: str, value: Any) -> None:
        with self._get_conn() as conn:
            self._ensure_agent(conn, agent_id)
            with conn.cursor() as cur:
                cur.execute("SELECT context FROM agent_memory WHERE agent_id = %s", (agent_id,))
                context = cur.fetchone()[0] or {}
                context[key] = value
                now = datetime.now()
                cur.execute("UPDATE agent_memory SET context = %s, updated_at = %s WHERE agent_id = %s",
                            (custom_dumps(context), now, agent_id))
            conn.commit()

    def get_context(self, agent_id: str) -> Dict:
        with self._get_conn() as conn:
            self._ensure_agent(conn, agent_id)
            with conn.cursor() as cur:
                cur.execute("SELECT context FROM agent_memory WHERE agent_id = %s", (agent_id,))
                return cur.fetchone()[0] or {}

    def save_strategy(self, name: str, strategy: Dict) -> str:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                now = datetime.now()
                cur.execute("""
                    INSERT INTO strategies (name, data, saved_at) 
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name) DO UPDATE 
                    SET data = EXCLUDED.data, saved_at = EXCLUDED.saved_at
                """, (name, custom_dumps(strategy), now))
            conn.commit()
        return f"db:strategies:{name}"

    def load_strategy(self, name: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("SELECT data, saved_at FROM strategies WHERE name = %s", (name,))
                row = cur.fetchone()
                if row:
                    data = dict(row["data"])
                    data["saved_at"] = row["saved_at"].isoformat()
                    return data
        return None

    def list_strategies(self) -> List[str]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM strategies")
                return [row[0] for row in cur.fetchall()]

    def log_trade(self, trade: Dict) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                now = datetime.now()
                today = now.date()
                symbol = trade.get('symbol', '').upper()
                if symbol.endswith("USDT") or symbol.startswith("P-") or symbol.startswith("C-") or symbol.startswith("F-"):
                    market_mode = 'crypto'
                elif symbol.endswith("=X"):
                    market_mode = 'forex'
                else:
                    market_mode = 'equity'
                
                # Extract normalized fields
                trade_type = (trade.get('transaction_type') or trade.get('trade_type') or 
                              trade.get('action') or trade.get('transaction') or '').upper()
                quantity = int(trade.get('quantity') or trade.get('qty') or 0)
                price = float(trade.get('price') or trade.get('average_price') or 0.0)
                pnl = trade.get('pnl')
                if pnl is not None:
                    pnl = float(pnl)
                
                cur.execute("""
                    INSERT INTO trade_logs (date, data, logged_at, market_mode, symbol, trade_type, quantity, price, pnl) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (today, custom_dumps(trade), now, market_mode, symbol, trade_type, quantity, price, pnl))

    def get_trade_log(self, date: Optional[str] = None, market_mode: Optional[str] = None) -> List[Dict]:
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                query = "SELECT data, logged_at FROM trade_logs WHERE 1=1"
                params = []
                if date:
                    query += " AND date = %s"
                    params.append(date)
                else:
                    today = datetime.now().date()
                    query += " AND date = %s"
                    params.append(today)
                
                if market_mode:
                    query += " AND market_mode = %s"
                    params.append(market_mode)
                    
                query += " ORDER BY logged_at ASC"
                cur.execute(query, tuple(params))
                
                rows = cur.fetchall()
                logs = []
                for r in rows:
                    log = dict(r["data"])
                    log["logged_at"] = r["logged_at"].isoformat()
                    logs.append(log)
                return logs

    def save_visualization_meta(self, name: str, meta: Dict) -> str:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                now = datetime.now()
                cur.execute("""
                    INSERT INTO visualizations (name, meta, saved_at) 
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name) DO UPDATE 
                    SET meta = EXCLUDED.meta, saved_at = EXCLUDED.saved_at
                """, (name, custom_dumps(meta), now))
            conn.commit()
        return f"db:visualizations:{name}"

    def list_visualizations(self) -> List[str]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM visualizations")
                return [row[0] for row in cur.fetchall()]
