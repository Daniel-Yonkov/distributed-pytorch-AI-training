#!/usr/bin/env python

import socket
from dotenv import load_dotenv
# from torchinfo import summary
from lib.socket import getTCPConnection,\
    startConnectionListenerProcess, sendWorldSizeAndEpochsToPeers
from lib.control import allPeersConnectedControlBlock
from lib.pytorch.main import run


load_dotenv()


def awaitClientsConnection(socketConnection: socket, epochs) -> int:
    connectionListenerProcess = startConnectionListenerProcess(
        socketConnection)
    allPeersConnectedControlBlock()
    worldSize = sendWorldSizeAndEpochsToPeers(epochs)
    connectionListenerProcess.terminate()
    return worldSize


if __name__ == "__main__":
    epochs = 10
    rank = 0  # server rank
    worldSize = awaitClientsConnection(getTCPConnection(), epochs)
    # Initialize the distributed environment.
    run(rank, worldSize, epochs)
