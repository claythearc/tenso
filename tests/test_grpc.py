import pytest
import numpy as np
import grpc
from concurrent import futures
import tenso
from tenso.grpc import tenso_msg_pb2, tenso_msg_pb2_grpc


# --- Mock Servicer ---
class TestServicer(tenso_msg_pb2_grpc.TensorInferenceServicer):
    def Predict(self, request, context):
        data = tenso.loads(request.tensor_packet)
        # Perform a simple operation to verify data is valid
        result = data * 2
        return tenso_msg_pb2.PredictResponse(
            result_packet=bytes(tenso.dumps(result)), status="SUCCESS"
        )


@pytest.fixture(scope="module")
def grpc_server():
    """Setup a local gRPC server for testing."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    tenso_msg_pb2_grpc.add_TensorInferenceServicer_to_server(TestServicer(), server)
    port = server.add_insecure_port("[::]:0")  # Random available port
    server.start()
    yield f"localhost:{port}"
    server.stop(0)


def test_grpc_roundtrip(grpc_server):
    """Verify that a tensor survives a full gRPC request/response cycle."""
    with grpc.insecure_channel(grpc_server) as channel:
        stub = tenso_msg_pb2_grpc.TensorInferenceStub(channel)

        original = np.random.rand(100, 100).astype(np.float32)
        packet = tenso.dumps(original)

        request = tenso_msg_pb2.PredictRequest(
            model_name="test_model", tensor_packet=bytes(packet)
        )

        response = stub.Predict(request)
        result = tenso.loads(response.result_packet)

        assert response.status == "SUCCESS"
        assert np.array_equal(result, original * 2)


def test_grpc_large_payload(grpc_server):
    """Verify that larger payloads (near 4MB) work correctly."""
    with grpc.insecure_channel(grpc_server) as channel:
        stub = tenso_msg_pb2_grpc.TensorInferenceStub(channel)

        # 1MB tensor
        data = np.zeros((512, 512), dtype=np.float32)
        request = tenso_msg_pb2.PredictRequest(tensor_packet=bytes(tenso.dumps(data)))

        response = stub.Predict(request)
        assert response.status == "SUCCESS"
