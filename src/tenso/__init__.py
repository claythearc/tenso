from .core import dumps, loads, dump, load, read_stream, write_stream
from .utils import get_packet_info, is_aligned

# Optional Async support
try:
    from .async_core import aread_stream
except ImportError:
    aread_stream = None

# Optional GPU support
try:
    from .gpu import read_to_device
except ImportError:
    read_to_device = None

__all__ = [
    "dumps", "loads", "dump", "load", 
    "read_stream", "write_stream", "aread_stream", "read_to_device",
    "get_packet_info", "is_aligned"
]