import socket
import numpy as np
import tenso
import time

# Connect
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 9999))

# 1. Create a dummy image (Batch of 4)
data = np.random.rand(4, 256, 256, 3).astype(np.float32)

print(f"Sending Tensor: {data.shape} ({data.nbytes / 1024 / 1024:.2f} MB)")

t0 = time.time()
# 2. Serialize and Send
packet = tenso.dumps(data)
client.sendall(packet)

# 3. Receive Result
# Read the response in chunks to handle large packets
response_data = b''
expected_size = len(packet)  # Response should be similar size
while len(response_data) < expected_size:
    chunk = client.recv(min(65536, expected_size - len(response_data)))
    if not chunk:
        break
    response_data += chunk

result = tenso.loads(response_data)

print(f"Got Result in {time.time() - t0:.4f}s")
print(f"Result Mean: {result.mean():.4f}")