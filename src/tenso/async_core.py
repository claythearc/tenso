import asyncio
import numpy as np
import struct
import math
from .core import loads, _REV_DTYPE_MAP, _MAGIC, _ALIGNMENT

async def aread_stream(reader: asyncio.StreamReader) -> np.ndarray:
    """
    Async zero-copy reader for FastAPI/asyncio servers.
    Reads directly from the stream into a pre-allocated numpy buffer.
    """
    # 1. Read Header
    try:
        header = await reader.readexactly(8)
    except asyncio.IncompleteReadError as e:
        # If we read 0 bytes, it's a graceful disconnect (EOF).
        # If we read partial bytes (1-7), it's a corrupted stream.
        if len(e.partial) == 0:
            return None
        raise e  # Re-raise for partial header

    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header)
    if magic != _MAGIC: raise ValueError("Invalid tenso packet")

    # 2. Read Shape
    shape_len = ndim * 4
    # readexactly raises IncompleteReadError automatically if stream ends here
    shape_bytes = await reader.readexactly(shape_len)
    shape = struct.unpack(f'<{ndim}I', shape_bytes)

    # 3. Calculate Layout
    dtype = _REV_DTYPE_MAP.get(dtype_code)
    if dtype is None: raise ValueError(f"Unknown dtype: {dtype_code}")

    current_pos = 8 + shape_len
    remainder = current_pos % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    body_len = int(math.prod(shape) * dtype.itemsize)
    total_len = current_pos + padding_len + body_len

    # 4. Allocate Buffer (Uninitialized)
    full_buffer = np.empty(total_len, dtype=np.uint8)
    
    # Fill Header/Shape (to make the buffer valid for loads)
    full_buffer[0:8] = list(header)
    full_buffer[8:8+shape_len] = list(shape_bytes)
    
    # 5. Read Body
    # We skip reading padding specifically since we can just read the body 
    # directly into the correct offset.
    if padding_len > 0:
        # We must consume the padding bytes from the stream
        await reader.readexactly(padding_len)

    # Read body directly into the buffer view
    # asyncio.readexactly returns bytes, so we have one copy here 
    # (unavoidable in pure asyncio without uvloop/manual socket handling)
    body_data = await reader.readexactly(body_len)
    
    body_start = current_pos + padding_len
    full_buffer[body_start:body_start+body_len] = np.frombuffer(body_data, dtype=np.uint8)

    return loads(full_buffer)