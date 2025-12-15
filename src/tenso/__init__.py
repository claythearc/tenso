from .core import dumps, loads, dump, load, read_stream, write_stream
from .utils import get_packet_info, is_aligned

# Optional Async support
try:
    from .async_core import aread_stream
except ImportError:
    aread_stream = None

__all__ = [
    "dumps", "loads", "dump", "load", 
    "read_stream", "write_stream", "aread_stream",
    "get_packet_info", "is_aligned"
]