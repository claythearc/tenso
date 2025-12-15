import pytest
import numpy as np
import tenso
import asyncio
import io

# Check if async_core is available
try:
    from tenso.async_core import aread_stream
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

@pytest.mark.skipif(not ASYNC_AVAILABLE, reason="Async core not implemented")
@pytest.mark.asyncio
async def test_aread_stream_success():
    """Test successful async read."""
    data = np.random.rand(20, 20).astype(np.float32)
    packet = tenso.dumps(data)
    
    # Mock asyncio StreamReader
    reader = asyncio.StreamReader()
    reader.feed_data(packet)
    reader.feed_eof()
    
    result = await aread_stream(reader)
    
    assert np.array_equal(data, result)

@pytest.mark.skipif(not ASYNC_AVAILABLE, reason="Async core not implemented")
@pytest.mark.asyncio
async def test_aread_stream_padding():
    """Test async read with padding."""
    # Force padding: 1 byte body
    data = np.array([123], dtype=np.uint8)
    packet = tenso.dumps(data)
    
    reader = asyncio.StreamReader()
    reader.feed_data(packet)
    # Feed extra data to ensure we don't over-read
    reader.feed_data(b'NEXT_REQ')
    reader.feed_eof()
    
    result = await aread_stream(reader)
    assert np.array_equal(data, result)
    
    # Verify next bytes are waiting
    remainder = await reader.read()
    assert remainder == b'NEXT_REQ'

@pytest.mark.skipif(not ASYNC_AVAILABLE, reason="Async core not implemented")
@pytest.mark.asyncio
async def test_aread_stream_incomplete():
    """Test async read disconnects."""
    data = np.zeros((10, 10), dtype=np.float32)
    packet = tenso.dumps(data)
    
    # 1. Incomplete Header
    reader = asyncio.StreamReader()
    reader.feed_data(packet[:4])
    reader.feed_eof()
    
    # Should return None if empty, or raise error if partial header?
    # Logic: readexactly raises IncompleteReadError
    with pytest.raises(asyncio.IncompleteReadError):
        await aread_stream(reader)

    # 2. Incomplete Body
    reader = asyncio.StreamReader()
    reader.feed_data(packet[:-5]) # Cut 5 bytes from body
    reader.feed_eof()
    
    with pytest.raises(asyncio.IncompleteReadError):
        await aread_stream(reader)