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
    HOST = '0.0.0.0'  # Listen on all interfaces
    PORT = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("Worker listening for connection...")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            # Receive chunk of A
            chunk_A = recv_matrix(conn)
            chunk_A = torch.tensor(chunk_A).cuda()

            # Receive full B
            B = recv_matrix(conn)
            B = torch.tensor(B).cuda()

            # Perform GPU matrix multiplication
            result = torch.matmul(chunk_A, B)

            # Send result back
            send_matrix(conn, result.cpu().numpy())

if __name__ == "__main__":
    main()
