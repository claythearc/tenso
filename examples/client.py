import socket
import numpy as np
import tenso
import time

# Connect
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("localhost", 9999))

# 1. Create a dummy image (Batch of 4)
data = np.random.rand(4, 256, 256, 3).astype(np.float32)

print(f"Sending Tensor: {data.shape} ({data.nbytes / 1024 / 1024:.2f} MB)")

t0 = time.time()
# 2. Serialize and Send (Zero-Copy Write)
packet = tenso.dumps(data)
client.sendall(packet)

# 3. Receive Result (Zero-Copy Read)
print("Receiving response...")
# Uses the new optimized reader (11.5 GB/s)
result = tenso.read_stream(client)

if result is not None:
    print(f"Got Result in {time.time() - t0:.4f}s")
    print(f"Result Shape: {result.shape} | Mean: {result.mean():.4f}")
else:
    print("Server disconnected.")

client.close()
