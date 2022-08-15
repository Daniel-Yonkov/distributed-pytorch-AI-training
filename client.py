import os
import socket
import ast
from lib.pytorch.main import run
from lib.socket import getMessage
from dotenv import load_dotenv

load_dotenv()


def connectToServer() -> tuple([int, int, int]):
    socketHost = os.environ['MASTER_ADDR']
    socketPort = os.environ['SERVER_PORT']
    s = socket.socket()
    print(f"[+] Connecting to {socketHost}:{socketPort}")
    s.connect((socketHost, int(socketPort)))
    print("[+] Connected.")
    rank = getMessage(s)
    worldSizeAndEpochs = ast.literal_eval(getMessage(s))
    worldSize = worldSizeAndEpochs['numberOfPeers']
    epochs = worldSizeAndEpochs['epochs']
    print("rank:", rank, "size:", worldSize, "epochs:", epochs)
    return (int(rank), int(worldSize), int(epochs))


if __name__ == "__main__":
    (rank, worldSize, epochs) = connectToServer()
    # Initialize the distributed environment.
    run(rank, worldSize, epochs)