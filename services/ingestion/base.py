# services/ingestion/base.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Signal:
    id: str
    source_type: str   # 'review' | 'ticket' | 'sales' | 'market' | 'internal'
    content: str
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    embedding: Optional[list] = None