import socket
import time
import os
from multiprocessing import Process, JoinableQueue

_socketConnection = None
_peerConnections = JoinableQueue()


def getTCPConnection() -> socket:
    global _socketConnection
    print("[-] Establishing TCP server...")
    # device's IP address
    SERVER_HOST = os.environ["SERVER_HOST"]
    SERVER_PORT = os.environ["SERVER_PORT"]

    print("[-] Awaiting client connections...")
    # create the server socket
    # TCP socket
    socketConnection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socketConnection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # bind the socket to our local address
    socketConnection.bind((SERVER_HOST, int(SERVER_PORT)))
    # enabling our server to accept connections
    # 5 here is the number of unaccepted connections that
    # the system will allow before refusing new connections
    socketConnection.listen(5)
    print("[*] Server connection established",
          f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}"
          )
    _socketConnection = socketConnection
    return socketConnection


def _listenForConnections(socketConnection: socket,) -> \
        tuple([socket, str, tuple([str, str])]):
    # accept connection if there is any
    client_socket, address = socketConnection.accept()
    rank = _storePeerConnection((client_socket, address))
    return (client_socket, rank, address)


def _storePeerConnection(
    connectionData: tuple([socket, tuple([str, str])])
) -> str:
    rank = _peerConnections.qsize() + 1
    socket, (ip, port) = connectionData
    peerConnection = {
        "socket": socket,
        "ip": ip,
        "rank": rank
    }
    _peerConnections.put(peerConnection)
    return rank


def _hasConnectionsToBeProcessed() -> bool:
    global _peerConnections
    return not _peerConnections.empty()


def _terminateClientConnection(con: socket):
    global _peerConnections
    _peerConnections.task_done()
    con.close()


def startConnectionListenerProcess(
    socketConnection: socket
) -> Process:
    p = Process(
        target=_connectionListener,
        args=[socketConnection]
    )
    p.daemon = True
    p.start()
    return p


def _connectionListener(
    socketConnection: socket,
) -> None:
    while True:
        socket, rank, (ip, port) = _listenForConnections(
            socketConnection)
        socket.sendall(str(rank).encode())
        print("[*] Rank:", rank, "connected with ip:", ip, 'and port:', port)


def sendWorldSizeAndEpochsToPeers(epochs) -> int:
    numberOfPeers = int(_getNumberOfPeers()) + 1  # server included
    while _hasConnectionsToBeProcessed():
        socket = _getPeerConnection()
        socket.sendall(
            str({"numberOfPeers": numberOfPeers, "epochs": epochs}).encode())
        _terminateClientConnection(socket)
    print("World size sent to all peers")
    print("Closing shared resources and TCP server")
    _closeSharedResources()
    _closeTCPConnection()
    print("Shared resources released and TCP server closed")
    return numberOfPeers


def _getNumberOfPeers() -> int:
    global _peerConnections
    return _peerConnections.qsize()


def _getPeerConnection() -> socket:
    global _peerConnections
    peerConnection = _peerConnections.get()
    return peerConnection['socket']


def _closeSharedResources() -> None:
    global _peerConnections
    _peerConnections.join()
    _peerConnections.close()


def _closeTCPConnection() -> None:
    global _socketConnection
    if(_socketConnection is not None):
        _socketConnection.close()
        _socketConnection = None
