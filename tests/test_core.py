import pytest
import numpy as np
import tenso

def test_alignment():
    """Ensure the body data starts at a 64-byte aligned offset."""
    # Create a weird shape to ensure header+shape isn't accidentally aligned
    # Header(8) + Shape(3 dims * 4 = 12) = 20 bytes. 
    # Needs 44 bytes of padding to reach 64.
    data = np.zeros((10, 10, 10), dtype=np.float32)
    packet = tenso.dumps(data)
    
    # Manually inspect the padding
    header_shape_len = 8 + (3 * 4)
    padding_len = 64 - header_shape_len
    
    # The bytes at the padding location should be zero
    padding_area = packet[header_shape_len : header_shape_len + padding_len]
    assert padding_area == b'\x00' * padding_len
    
    # Verify load works
    restored = tenso.loads(packet)
    assert np.array_equal(data, restored)

def test_safety_copy():
    """Ensure copy=True creates a distinct memory object."""
    data = np.array([1, 2, 3, 4], dtype=np.float32)
    packet = tenso.dumps(data)
    
    # Zero-Copy (Default)
    view = tenso.loads(packet, copy=False)
    # View should point to packet's memory (simplified check)
    assert view.base is not None 

    # Forced Copy
    safe_copy = tenso.loads(packet, copy=True)
    # Copy should own its own memory (base is None)
    assert safe_copy.base is None
    
    assert np.array_equal(view, safe_copy)

def test_non_contiguous_input():
    """Ensure sliced arrays are handled automatically."""
    # Create non-contiguous array via slicing
    matrix = np.random.rand(10, 10).astype(np.float32)
    sliced = matrix[:, ::2] # Stride of 2
    assert not sliced.flags['C_CONTIGUOUS']
    
    packet = tenso.dumps(sliced)
    restored = tenso.loads(packet)
    
    assert np.array_equal(sliced, restored)
    assert restored.flags['C_CONTIGUOUS'] # Result is always contiguous

def test_version_check():
    """Ensure we catch version mismatches if future versions arise."""
    data = np.zeros((2,2), dtype=np.float32)
    packet = bytearray(tenso.dumps(data))
    
    # Tamper with version byte (set to 99)
    # Header format: <4sBBBB -> Magic(4), Ver(1), Flags(1)...
    # Version is at index 4
    packet[4] = 99 
    
    with pytest.raises(ValueError, match="Unsupported version"):
        tenso.loads(bytes(packet))