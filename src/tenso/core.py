import struct
import numpy as np

# --- The Tenso Protocol v2 ---
_MAGIC = b'TNSO'
_VERSION = 2  # Bumping version because format changed (Padding)
_ALIGNMENT = 64  # Align body to 64-byte boundaries for AVX-512/SIMD

# Dtype Mapping (Includes uint8, float16, etc.)
_DTYPE_MAP = {
    np.dtype('float32'): 1,
    np.dtype('int32'): 2,
    np.dtype('float64'): 3,
    np.dtype('int64'): 4,
    np.dtype('uint8'): 5,
    np.dtype('uint16'): 6,
    np.dtype('bool'): 7,
    np.dtype('float16'): 8,
}
_REV_DTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items()}

def dumps(tensor: np.ndarray) -> bytes:
    """
    Serialize a numpy array into bytes with 64-byte alignment.
    """
    # 1. Validation & Preparation
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    # Force C-Contiguous layout (Safety Check #3)
    # If it's already contiguous, this does nothing. If not, it creates a copy.
    if not tensor.flags['C_CONTIGUOUS']:
        tensor = np.ascontiguousarray(tensor)

    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    # 2. Calculate Sizes for Alignment
    header_size = 8
    shape_size = ndim * 4
    current_offset = header_size + shape_size
    
    # Calculate Padding needed to reach next 64-byte boundary
    remainder = current_offset % _ALIGNMENT
    padding_size = 0 if remainder == 0 else _ALIGNMENT - remainder
    
    # 3. Construct Parts
    # Flags=1 indicates "Aligned 64" logic is active
    header = struct.pack('<4sBBBB', _MAGIC, _VERSION, 1, dtype_code, ndim)
    shape_block = struct.pack(f'<{ndim}I', *shape)
    padding = b'\x00' * padding_size
    
    # 4. Zero-Copy assembly (tobytes creates a view or copy depending on memory)
    return header + shape_block + padding + tensor.tobytes()

def loads(data: bytes, copy: bool = False) -> np.ndarray:
    """
    Deserialize bytes back into a numpy array.
    
    Args:
        data: The bytes object containing the Tenso packet.
        copy: If True, forces a memory copy. Safer if 'data' is a temporary buffer
              that might be freed (e.g., from a socket loop).
              If False (default), returns a Zero-Copy view (Fastest).
    """
    # 1. Basic Size Check
    if len(data) < 8:
        raise ValueError("Packet too short to contain header")

    # 2. Parse Header
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', data[:8])
    
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet (Magic bytes mismatch)")
    
    # Version Compatibility Check
    if ver > _VERSION:
        raise ValueError(f"Unsupported version: {ver} (Library supports v{_VERSION})")

    # 3. Parse Shape
    shape_start = 8
    shape_end = 8 + (ndim * 4)
    
    if len(data) < shape_end:
        raise ValueError("Packet too short to contain shape data")
        
    shape = struct.unpack(f'<{ndim}I', data[shape_start:shape_end])
    
    # 4. Handle Alignment / Padding
    # v2 packets (and flags=1) have padding. v1 did not.
    body_start = shape_end
    if ver >= 2:
        remainder = shape_end % _ALIGNMENT
        padding_size = 0 if remainder == 0 else _ALIGNMENT - remainder
        body_start += padding_size

    # 5. Validate Body Size (Safety Check #2)
    dtype = _REV_DTYPE_MAP[dtype_code]
    expected_body_size = np.prod(shape) * dtype.itemsize
    
    # We allow the packet to be larger (e.g. extra buffer at end), but not smaller
    if len(data) < body_start + expected_body_size:
        raise ValueError("Packet too short (Body truncated)")

    # 6. Create Array
    # 'frombuffer' is the Zero-Copy magic
    arr = np.frombuffer(data, dtype=dtype, offset=body_start, count=np.prod(shape))
    arr = arr.reshape(shape)

    # 7. Handle Safety Copy (Safety Check #1)
    if copy:
        return arr.copy()
    
    return arr

# File I/O Helpers
def dump(tensor: np.ndarray, fp) -> None:
    fp.write(dumps(tensor))

def load(fp) -> np.ndarray:
    return loads(fp.read())