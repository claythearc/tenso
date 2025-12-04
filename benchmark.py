import time
import json
import pickle
import io
import msgpack
import numpy as np
import tenso
from safetensors.numpy import save as st_save, load as st_load
import pyarrow as pa

# --- CONFIGURATION ---
ITERATIONS = 20  # Reduced slightly as these libraries are heavy

# --- COMPETITORS ---
def bench_json(data):
    # JSON is too slow for large data, we wrap it to prevent crashes
    enc = lambda x: json.dumps(x.tolist()).encode('utf-8')
    dec = lambda x: np.array(json.loads(x), dtype=data.dtype)
    return enc, dec

def bench_pickle(data):
    enc = lambda x: pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)
    dec = lambda x: pickle.loads(x)
    return enc, dec

def bench_msgpack(data):
    enc = lambda x: msgpack.packb(x.tobytes())
    dec = lambda x: np.frombuffer(msgpack.unpackb(x), dtype=data.dtype).reshape(data.shape)
    return enc, dec

def bench_safetensors(data):
    # Safetensors expects a dictionary of tensors, we wrap our single tensor
    def enc(x):
        return st_save({"t": x})
    def dec(x):
        return st_load(x)["t"]
    return enc, dec

def bench_arrow(data):
    # Arrow works best with Tables/RecordBatches
    def enc(x):
        # Flatten because Arrow is columnar (1D usually)
        arr = pa.array(x.flatten())
        batch = pa.RecordBatch.from_arrays([arr], names=['t'])
        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        return sink.getvalue()
    
    def dec(x):
        reader = pa.ipc.open_stream(x)
        batch = reader.read_next_batch()
        # Zero-copy conversion back to numpy
        np_arr = batch.column(0).to_numpy(zero_copy_only=False) 
        return np_arr.reshape(data.shape)
    return enc, dec

def bench_tenso(data):
    return tenso.dumps, tenso.loads

# --- SCENARIOS ---
SCENARIOS = [
    {
        "name": "API Vector",
        "desc": "Small 1D Embedding",
        "shape": (1536,), 
        "dtype": np.float32
    },
    {
        "name": "CV Batch",
        "desc": "32x 256x256 Images",
        "shape": (32, 256, 256, 3), 
        "dtype": np.uint8
    },
    {
        "name": "LLM Layer",
        "desc": "4096^2 Matrix",
        "shape": (4096, 4096), 
        "dtype": np.float32
    }
]

# --- RUNNER ---
print(f"{'SCENARIO':<15} | {'FORMAT':<12} | {'SIZE':<10} | {'SERIALIZE':<10} | {'DESERIALIZE':<10}")
print("-" * 75)

for scen in SCENARIOS:
    if scen['dtype'] == np.uint8:
        data = np.random.randint(0, 255, scen['shape']).astype(np.uint8)
    else:
        data = np.random.rand(*scen['shape']).astype(scen['dtype'])
    
    competitors = {
        "Pickle": bench_pickle(data),
        "MsgPack": bench_msgpack(data),
        "Safetensors": bench_safetensors(data),
        "Arrow": bench_arrow(data),
        "Tenso": bench_tenso(data)
    }

    results = {}

    for name, (enc_func, dec_func) in competitors.items():
        try:
            # Warmup
            encoded = enc_func(data)
            _ = dec_func(encoded)
            
            # Measure Ser
            t0 = time.perf_counter()
            for _ in range(ITERATIONS):
                encoded = enc_func(data)
            t_ser = ((time.perf_counter() - t0) / ITERATIONS) * 1000

            # Measure Des
            t0 = time.perf_counter()
            for _ in range(ITERATIONS):
                _ = dec_func(encoded)
            t_des = ((time.perf_counter() - t0) / ITERATIONS) * 1000
            
            size_str = f"{len(encoded)/1024/1024:.2f} MB" if len(encoded) > 1024*1024 else f"{len(encoded)/1024:.2f} KB"
            
            print(f"{scen['name']:<15} | {name:<12} | {size_str:<10} | {t_ser:>7.3f} ms | {t_des:>7.3f} ms")
            
        except Exception as e:
            print(f"{scen['name']:<15} | {name:<12} | FAILED ({e})")

    print("-" * 75)