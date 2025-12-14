import socket
import numpy as np
import tenso

# --- SERVER ---
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 9999))
server.listen(1)

print("Tenso Inference Server Waiting...")

conn, addr = server.accept()
print(f"Connected by {addr}")

while True:
    try:
        # Use the new core function instead of manual parsing
        tensor = tenso.read_stream(conn)
        
        # If None, client disconnected gracefully
        if tensor is None: 
            break
        
        print(f"Received Input: {tensor.shape} | Mean: {tensor.mean():.4f}")
        
        # Simulate Inference (dummy processing)
        result = tensor * 2 
        
        # Send result back
        print("Sending response...")
        conn.sendall(tenso.dumps(result))
        
    except Exception as e:
        print(f"Error: {e}")
        break

conn.close()