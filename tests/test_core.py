import pytest
import numpy as np
import tenso
import struct

# --- Core Functionality Tests ---

@pytest.mark.parametrize("dtype", [
    np.float32, np.int32, np.float64, np.int64,
    np.uint8, np.uint16, np.bool_, np.float16,
    np.int8, np.int16, np.uint32, np.uint64
])
def test_all_dtypes(dtype):
    """Verify all supported dtypes serialize and deserialize correctly."""
    shape = (10, 10)
    if dtype == np.bool_:
        original = np.random.randint(0, 2, size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        original = np.random.randint(0, 10, size=shape).astype(dtype)
    else:
        original = np.random.randn(*shape).astype(dtype)
        
    packet = tenso.dumps(original)
    restored = tenso.loads(packet)
    
    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    assert np.array_equal(original, restored)

def test_large_dimensions():
    """Verify handling of high-dimensional arrays."""
    # Create a 6D array (max ndim is 255 in protocol v2)
    shape = (2, 2, 2, 2, 2, 2)
    original = np.zeros(shape, dtype=np.float32)
    
    packet = tenso.dumps(original)
    restored = tenso.loads(packet)
    
    assert restored.shape == shape
    
    # Test packet inspection
    info = tenso.get_packet_info(packet)
    assert info['ndim'] == 6
    assert info['shape'] == shape

# --- Introspection Tests ---

def test_get_packet_info():
    """Verify introspection returns correct metadata without deserializing."""
    data = np.random.rand(32, 128).astype(np.float32)
    packet = tenso.dumps(data)
    
    info = tenso.get_packet_info(packet)
    
    assert info['version'] == 2
    assert info['dtype'] == np.dtype('float32')
    assert info['shape'] == (32, 128)
    assert info['ndim'] == 2
    assert info['aligned'] is True
    assert info['total_elements'] == 32 * 128
    assert info['data_size_bytes'] == 32 * 128 * 4  # float32 is 4 bytes

def test_packet_info_invalid():
    """Verify introspection raises errors on bad data."""
    with pytest.raises(ValueError, match="Packet too short"):
        tenso.get_packet_info(b'short')
        
    with pytest.raises(ValueError, match="Invalid tenso packet"):
        tenso.get_packet_info(b'JUNK____')

# --- Alignment & Safety Tests ---

def test_is_aligned_utility():
    """Test the is_aligned utility function."""
    # Note: We can't easily force the allocator to give us aligned/unaligned 
    # addresses in pure Python tests, so we mainly check it runs and returns bool.
    data = b'\x00' * 100
    aligned = tenso.is_aligned(data)
    assert isinstance(aligned, bool)

def test_padding_logic():
    """Verify that packets contain correct padding for 64-byte alignment."""
    # 1D array of float32
    # Header (8) + Shape (1*4 = 4) = 12 bytes offset.
    # Needs 52 bytes padding to reach 64.
    data = np.zeros((10,), dtype=np.float32)
    packet = tenso.dumps(data)
    
    header_shape_len = 8 + 4
    padding_len = 64 - header_shape_len
    
    # Extract padding area directly
    padding = packet[header_shape_len : header_shape_len + padding_len]
    assert len(padding) == padding_len
    assert padding == b'\x00' * padding_len

def test_copy_flag():
    """Verify copy=True behavior."""
    data = np.zeros((10,), dtype=np.float32)
    packet = tenso.dumps(data)
    
    # Default (Zero Copy)
    arr_view = tenso.loads(packet, copy=False)
    # Writeable flag should be False for zero-copy views to prevent corruption
    assert arr_view.flags.writeable is False
    
    # Copy (Safe)
    arr_copy = tenso.loads(packet, copy=True)
    assert arr_copy.flags.writeable is True
    assert arr_copy.base is None  # Owns its memory