import os
import socket
import time
import ast
from lib.pytorch.main import run
from dotenv import load_dotenv

load_dotenv()


BUFFER_SIZE = 4096  # send 4096 bytes each time step


def connectToServer() -> socket:
    socketHost = os.environ['MASTER_ADDR']
    socketPort = os.environ['SERVER_PORT']
    s = socket.socket()
    print(f"[+] Connecting to {socketHost}:{socketPort}")
    s.connect((socketHost, int(socketPort)))
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
    run(rank, worldSize, epochs)
    end = time.time()

    print("Time to iterate trought", epochs, "epochs:", end - start)
