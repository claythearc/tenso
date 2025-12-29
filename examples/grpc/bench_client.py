import grpc
import numpy as np
import tenso
import time
from tenso.grpc import benchmark_msg_pb2
from tenso.grpc import benchmark_msg_pb2_grpc


def run_bench():
    # Increase message limits for the client channel
    MAX_MESSAGE_LENGTH = 128 * 1024 * 1024
    options = [
        ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
    ]

    channel = grpc.insecure_channel("localhost:50051", options=options)
    stub = benchmark_msg_pb2_grpc.BenchmarkerStub(channel)

    # Create a 4MB tensor (1024x1024 float32)
    data = np.random.rand(1024, 1024).astype(np.float32)
    iterations = 50

    print(
        f"Benchmarking {iterations} iterations of {data.nbytes / 1024 / 1024:.2f} MB tensor..."
    )

    # --- Test 1: Tenso + gRPC ---
    print("Starting Tenso Test...")
    t0 = time.perf_counter()
    for _ in range(iterations):
        packet = tenso.dumps(data)
        stub.SendTenso(benchmark_msg_pb2.TensoRequest(data=bytes(packet)))
    t_tenso = time.perf_counter() - t0
    avg_tenso = (t_tenso / iterations) * 1000
    print(f"Tenso Results:   {t_tenso:.4f}s total | Avg: {avg_tenso:.2f} ms/op")

    # --- Test 2: Standard Protobuf ---
    print("Starting Standard gRPC Test (This will be slow)...")
    t0 = time.perf_counter()
    flat_data = data.flatten().tolist()

    for _ in range(iterations):
        # Protobuf requires a flat Python list
        stub.SendStandard(benchmark_msg_pb2.StandardRequest(data=flat_data))
    t_std = time.perf_counter() - t0
    avg_std = (t_std / iterations) * 1000
    print(f"Standard Results: {t_std:.4f}s total | Avg: {avg_std:.2f} ms/op")

    print(f"\nConclusion: Tenso is {t_std / t_tenso:.1f}x faster than Standard gRPC")


if __name__ == "__main__":
    run_bench()
