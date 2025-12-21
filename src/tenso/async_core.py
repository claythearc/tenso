import asyncio
import numpy as np
import struct
import math
from typing import Union
from .core import loads, _REV_DTYPE_MAP, _MAGIC, _ALIGNMENT


async def aread_stream(reader: asyncio.StreamReader) -> Union[np.ndarray, None]:
    """
    True Zero-Copy Async Read: Eliminates intermediate buffer copies.
    """
    try:
        header = await reader.readexactly(8)
    except asyncio.IncompleteReadError as e:
        if len(e.partial) == 0: return None
        raise

    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header)
    
    # Read Shape
    shape_bytes = await reader.readexactly(ndim * 4)
    shape = struct.unpack(f'<{ndim}I', shape_bytes)
    
    # Calculate Padding
    current_pos = 8 + (ndim * 4)
    remainder = current_pos % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    # Consume Padding (Consistently discarded)
    if padding_len > 0:
        await reader.readexactly(padding_len)

    # Read Body
    dtype = _REV_DTYPE_MAP[dtype_code]
    body_len = int(np.prod(shape) * dtype.itemsize)
    body_data = await reader.readexactly(body_len)
    
    # Optimization: np.frombuffer on 'bytes' is zero-copy in Python
    # This avoids copying body_data into a pre-allocated full_buffer.
    arr = np.frombuffer(body_data, dtype=dtype).reshape(shape)
    arr.flags.writeable = False 
    return arr