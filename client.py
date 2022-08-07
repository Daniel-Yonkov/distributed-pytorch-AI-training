import socket
import time
import ast
from lib.pytorch.main import run
from dotenv import load_dotenv

load_dotenv()


BUFFER_SIZE = 4096  # send 4096 bytes each time step

# the ip address or hostname of the server, the receiver,
host = 'server'
port = 5001


def connectToServer() -> socket:
    global host, port, __ID
    s = socket.socket()
    print(f"[+] Connecting to {host}:{port}")
    s.connect((host, port))
    print("[+] Connected.")
    rank = getMessage(s)
    worldSizeAndEpochs = ast.literal_eval(getMessage(s))
    print(worldSizeAndEpochs)
    print("rank:", rank, "size:", worldSizeAndEpochs['numberOfPeers'])
    return (int(rank), int(worldSizeAndEpochs['numberOfPeers']), int(worldSizeAndEpochs['epochs']))


def getMessage(connection: socket) -> int:
    global BUFFER_SIZE
    return connection.recv(BUFFER_SIZE).decode()


if __name__ == "__main__":
    (rank, worldSize, epochs) = connectToServer()
    """ Initialize the distributed environment. """
    # TODO extract into config
    start = time.time()
    print(type(rank), type(worldSize), type(epochs))
    run(rank, worldSize, epochs)
    end = time.time()

    print("Time to iterate trought", epochs, "epochs:", end - start)