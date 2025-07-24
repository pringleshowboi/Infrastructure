import socket
import pickle
import torch
import numpy as np

def send_matrix(sock, matrix):
    data = pickle.dumps(matrix)
    sock.sendall(len(data).to_bytes(8, 'big'))
    sock.sendall(data)

def recv_matrix(sock):
    length = int.from_bytes(sock.recv(8), 'big')
    data = b''
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            raise ConnectionError("Socket connection lost")
        data += packet
    matrix = pickle.loads(data)
    return matrix

def main():
    HOST = '127.0.0.1'  # Change to worker's IP if remote
    PORT = 65432

    # Create large matrices
    A = torch.randn(1000, 500).cuda()  # Big matrix A
    B = torch.randn(500, 600).cuda()   # Big matrix B

    # Split A into two chunks (simulate 2 workers)
    chunks = torch.chunk(A, 2, dim=0)

    # Create socket and connect to worker
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Connected to worker")

        # Send chunk 0 of A and full B to worker
        send_matrix(s, chunks[0].cpu().numpy())
        send_matrix(s, B.cpu().numpy())

        # Receive result from worker
        result_chunk0 = recv_matrix(s)
        result_chunk0 = torch.tensor(result_chunk0).cuda()

        # Do chunk 1 locally (simulate second worker on master)
        result_chunk1 = torch.matmul(chunks[1], B)

        # Combine results
        final_result = torch.cat([result_chunk0, result_chunk1], dim=0)

        print("Distributed matrix multiplication result shape:", final_result.shape)

if __name__ == "__main__":
    main()
