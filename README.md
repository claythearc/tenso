<!-- <img width="2816" height="1536" alt="Gemini_Generated_Image_v39t46v39t46v39t" src="https://github.com/user-attachments/assets/50378dc3-6165-4b79-831d-5ebf1303cada" /> -->
<img width="2439" height="966" alt="Gemini_Generated_Image_v39t46v39t46v39t" src="https://github.com/user-attachments/assets/5ec9b225-3615-4225-82ca-68e15b7045ce" />

# Tenso

High-Performance, Zero-Copy Tensor Protocol for Python.

## Overview

Tenso is a specialized binary protocol designed for one thing: moving NumPy arrays between backends instantly.

It avoids the massive CPU overhead of standard formats (JSON, Pickle, MsgPack) by using a strict Little-Endian, 64-byte aligned memory layout. This allows for Zero-Copy deserialization, meaning the CPU doesn't have to move data—it just points to it.

### The Zero-CPU Advantage

Tenso isn't just about speed; it's about resource efficiency.

- JSON/Pickle: Parsing large arrays consumes significant CPU cycles (100% usage during load). In a high-throughput cluster, this steals resources from your actual model inference.

- Tenso: Deserialization is effectively 0% CPU. The processor simply maps the existing memory address.

## Benchmark

**Scenario 1: Large Matrix (64MB Float32)**
Reading from memory/disk.

| Format | Read Time | Throughput | Status |
| :--- | :--- | :--- | :--- |
| **Tenso** | **0.003 ms** | **21 GB/s** | **Instant** |
| **Arrow** | 0.009 ms | 7.1 GB/s | Fast |
| **Pickle** | 3.097 ms | 20 MB/s | Slow |

**Scenario 2: Stream Throughput (95MB Packet)**
Reading from a continuous data stream.

| Method | Time | Throughput |
| :--- | :--- | :--- |
| **Tenso (read_stream)** | **18.78 ms** | **5,078 MB/s** |
| **Standard Loop** | 6552.08 ms | 14 MB/s |

**Scenario 3: High-Frequency Stream (1KB Tensor)**
Writing to a socket (simulating low-latency robotics/gaming).

| Method | Writes/Sec | Latency |
| :--- | :--- | :--- |
| **Tenso (write_stream)** | **198,448** | **5.0 μs** |



## Installation

```bash
pip install tenso
```

## Usage

### Network (Sockets & Pipes)

```python
import numpy as np
import tenso
import socket

# --- SENDER (Client) ---
# Uses os.writev for atomic, single-syscall writes
data = np.random.rand(100, 100).astype(np.float32)
tenso.write_stream(data, client_socket)

# --- RECEIVER (Server) ---
# Uses readinto() for zero-allocation buffering
tensor = tenso.read_stream(conn)

if tensor is None:
    print("Connection closed")
```

### Basic Serialisation

```python
import numpy as np
import tenso

# Create a tensor
data = np.random.rand(100, 100).astype(np.float32)

# Serialize to bytes
packet = tenso.dumps(data)

# Deserialize back
restored = tenso.loads(packet)
```

### Network

```python
import numpy as np
import tenso

# Create a tensor
data = np.random.rand(100, 100).astype(np.float32)

# Serialize to bytes
packet = tenso.dumps(data)

# Deserialize back
restored = tenso.loads(packet)
```

### File I/O
```python
# Load from disk (Standard)
with open("weights.tenso", "rb") as f:
    loaded_data = tenso.load(f)

# Load Large Models (Larger than RAM)
# Uses OS memory mapping to read data instantly without loading file into memory
with open("llama_70b_weights.tenso", "rb") as f:
    loaded_data = tenso.load(f, mmap_mode=True)
```

## Protocol Specification

Tenso uses a Hybrid Fixed-Header format designed for SIMD safety.

- Header (8 bytes): TNSO Magic, Version, Flags, Dtype, NDim.

- Shape Block: Variable length (NDim * 4 bytes).

- Padding: 0-63 bytes to ensure the Body starts at a 64-byte aligned address.

- Body: Raw C-contiguous memory dump.

### Tenso vs. The World

| Feature | Tenso | Pickle | Arrow | Safetensors |
| :--- | :--- | :--- | :--- | :--- |
| **Speed (Read)** | **Instant** | Slow | Instant | Fast |
| **Safety** | **Secure** | Unsafe (RCE Risk) | Secure | Secure |
| **Alignment** | **64-byte** | None | 64-byte | None |
| **Dependencies** | **NumPy Only** | Python | PyArrow (Huge) | Rust/Bindings |
| **Best For** | **Network/IPC** | Python Objects | Dataframes | Disk Storage |

Why is Tenso 1500x faster than Pickle? Standard formats must copy data from the network buffer into a new NumPy array. Tenso uses Memory Mapping: it tells NumPy to point directly at the existing buffer. No copying, no CPU cycles.

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/tenso.git
cd tenso

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run benchmarks
python benchmark.py
```

### Advanced Usage

**Strict Mode:**
Prevent accidental memory copies during serialization. Raises an error if data is not already C-Contiguous.

```python
try:
    # Will raise ValueError if array is Fortran-contiguous or non-contiguous
    packet = tenso.dumps(data, strict=True) 
except ValueError:
    print("Array must be C-Contiguous!")

```

## Requirements

- Python >= 3.10
- NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
