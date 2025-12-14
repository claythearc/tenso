import struct
import numpy as np
from typing import BinaryIO, Union, Any
import math
import mmap
import sys
from .config import _MAGIC, _VERSION, _ALIGNMENT, _DTYPE_MAP, _REV_DTYPE_MAP

IS_LITTLE_ENDIAN = (sys.byteorder == 'little')

# --- Stream Helper ---

def _read_exact(source: Any, n: int) -> Union[bytes, None]:
    """Optimized reader that avoids string concatenation."""
    if n == 0:
        return b''
        
    # Pre-allocate buffer (Zero allocation during loop)
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    
    while pos < n:
        if hasattr(source, 'recv_into'):
            # Optimized Socket Read
            bytes_read = source.recv_into(view[pos:])
        elif hasattr(source, 'readinto'):
             # Optimized File Read
            bytes_read = source.readinto(view[pos:])
        else:
            # Fallback: Object has no zero-copy methods
            # Check for 'recv' (Socket) or 'read' (File)
            if hasattr(source, 'recv'):
                chunk = source.recv(n - pos)
            else:
                chunk = source.read(n - pos)
            
            if not chunk:
                bytes_read = 0
            else:
                # Copy chunk into our pre-allocated buffer
                view[pos:pos+len(chunk)] = chunk
                bytes_read = len(chunk)

        if bytes_read == 0:
            if pos == 0: return None
            raise EOFError(f"Stream closed. Expected {n} bytes, got {pos}")
            
        pos += bytes_read
        
    return bytes(buf)


def read_stream(source: Any) -> Union[np.ndarray, None]:
    """
    Reads a complete Tenso packet directly from a socket or file stream.
    """
    # 1. Read Fixed Header
    try:
        header = _read_exact(source, 8)
    except EOFError as e:
        raise EOFError("Stream ended during header read") from e
        
    if header is None:
        return None
        
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header)
    
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet (magic bytes mismatch)")

    # 2. Read Shape Block
    shape_len = ndim * 4
    try:
        shape_bytes = _read_exact(source, shape_len)
    except EOFError as e:
        raise EOFError("Stream ended during shape read") from e
    
    shape = struct.unpack(f'<{ndim}I', shape_bytes)
    
    # 3. Calculate Padding
    current_offset = 8 + shape_len
    remainder = current_offset % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    # 4. Read Padding
    try:
        padding = _read_exact(source, padding_len)
    except EOFError as e:
        raise EOFError("Stream ended during padding read") from e
    if padding is None: padding = b''

    # 5. Calculate Body Size
    dtype = _REV_DTYPE_MAP.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unknown dtype code: {dtype_code}")
        
    total_elements = math.prod(shape)
    body_len = int(total_elements * dtype.itemsize)
    
    # 6. Read Body
    try:
        body = _read_exact(source, body_len)
    except EOFError as e:
        raise EOFError("Stream ended during body read") from e
    if body is None: body = b''

    return loads(header + shape_bytes + padding + body)


# --- Core Functions ---

def dumps(tensor: np.ndarray, strict: bool = False) -> bytes:
    """Serialize a numpy array into bytes with 64-byte alignment."""
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    if not tensor.flags['C_CONTIGUOUS']:
        if strict:
            raise ValueError("Tensor is not C-Contiguous and strict=True.")
        tensor = np.ascontiguousarray(tensor)

    if not IS_LITTLE_ENDIAN or tensor.dtype.byteorder == '>':
        tensor = tensor.astype(tensor.dtype.newbyteorder('<'))

    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    if ndim > 255:
        raise ValueError(f"Too many dimensions: {ndim} (max 255)")
    
    header_size = 8
    shape_size = ndim * 4
    current_offset = header_size + shape_size
    
    remainder = current_offset % _ALIGNMENT
    padding_size = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    header = struct.pack('<4sBBBB', _MAGIC, _VERSION, 1, dtype_code, ndim)
    shape_block = struct.pack(f'<{ndim}I', *shape)
    padding = b'\x00' * padding_size
    
    return header + shape_block + padding + tensor.tobytes()


def loads(data: Union[bytes, mmap.mmap], copy: bool = False) -> np.ndarray:
    """Deserialize bytes back into a numpy array."""
    if len(data) < 8:
        raise ValueError("Packet too short to contain header")
    
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', data[:8])
    
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet")
    
    if ver > _VERSION:
        raise ValueError(f"Unsupported version: {ver}")
    
    if dtype_code not in _REV_DTYPE_MAP:
        raise ValueError(f"Unknown dtype code: {dtype_code}")
    
    shape_start = 8
    shape_end = 8 + (ndim * 4)
    shape = struct.unpack(f'<{ndim}I', data[shape_start:shape_end])
    
    body_start = shape_end
    if ver >= 2 and flags & 1:
        remainder = shape_end % _ALIGNMENT
        padding_size = 0 if remainder == 0 else (_ALIGNMENT - remainder)
        body_start += padding_size
    
    dtype = _REV_DTYPE_MAP[dtype_code]
    
    arr = np.frombuffer(
        data,
        dtype=dtype,
        offset=body_start,
        count=int(np.prod(shape))
    )
    arr = arr.reshape(shape)
    
    if copy: return arr.copy()
    
    arr.flags.writeable = False
    return arr


def dump(tensor: np.ndarray, fp: BinaryIO, strict: bool = False) -> None:
    """
    Serialize a numpy array to a file-like object using Coalesced Writes.
    """
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    if not tensor.flags['C_CONTIGUOUS']:
        if strict:
            raise ValueError("Tensor is not C-Contiguous and strict=True.")
        tensor = np.ascontiguousarray(tensor)

    if not IS_LITTLE_ENDIAN or tensor.dtype.byteorder == '>':
        tensor = tensor.astype(tensor.dtype.newbyteorder('<'))

    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    if ndim > 255:
        raise ValueError(f"Too many dimensions: {ndim}")
    
    header_size = 8
    shape_size = ndim * 4
    current_offset = header_size + shape_size
    
    remainder = current_offset % _ALIGNMENT
    padding_size = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    # Optimization: Write header+shape+padding in one go
    header = struct.pack('<4sBBBB', _MAGIC, _VERSION, 1, dtype_code, ndim)
    shape_block = struct.pack(f'<{ndim}I', *shape)
    padding = b'\x00' * padding_size
    
    fp.write(header + shape_block + padding)
    fp.write(tensor.data)


def load(fp: BinaryIO, mmap_mode: bool = False, copy: bool = False) -> np.ndarray:
    """
    Deserialize a numpy array from a file-like object.
    """
    if mmap_mode:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        return loads(mm, copy=copy)
    else:
        return loads(fp.read(), copy=copy)