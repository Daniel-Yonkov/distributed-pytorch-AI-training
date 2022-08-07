import socket
import os
from multiprocessing import Process, JoinableQueue

_dbConnection = None
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


def listenForConnections(socketConnection: socket,) -> None:
    # accept connection if there is any
    client_socket, address = socketConnection.accept()
    # if below code is executed, that means the sender is connected
    print(f"[+] {address} is connected.")
    rank = storePeerConnection((client_socket, address))
    return (client_socket, rank, address)


def storePeerConnection(
    connectionData: tuple([socket, str])
) -> None:
    rank = _peerConnections.qsize() + 1
    (ip, port) = connectionData[1]
    peerConnection = {
        "socket": connectionData[0],
        "ip": ip,
        "rank": rank
    }
    _peerConnections.put(peerConnection)
    return rank


def hasConnectionsToBeProcessed() -> bool:
    global _peerConnections
    return not _peerConnections.empty()


def _finishConnectingClient(con: socket):
    global _peerConnections
    _peerConnections.task_done()
    con.close()


def finishConnectionProcess():
    global _peerConnections
    _peerConnections.join()
    _closeTCPConnection()


def getMessage(connection: socket):
    BUFFER_SIZE = 4096
    peerId = connection.recv(BUFFER_SIZE).decode()

    file = bytearray()
    # Workaround to hanging of socket.recv
    connection.settimeout(0.01)
    while True:
        try:
            data = connection.recv(BUFFER_SIZE)
            file.extend(data)
        except socket.timeout:
            if len(file) > 0:
                break

    return (peerId, file)


def startConnectionListenerProcess(
    socketConnection: socket
) -> Process:
    global _peerConnections
    p = Process(
        target=connectionListenerProcess,
        args=[socketConnection]
    )
    p.daemon = True
    p.start()
    return p


def connectionListenerProcess(
    socketConnection: socket,
) -> None:
    while True:
        socket, rank, (ip, port) = listenForConnections(
            socketConnection)
        socket.sendall(str(rank).encode())
        print("[*] Rank:", rank, "connected with ip:", ip, 'and port:', port)


def sendWorldSizeAndEpochsToPeers(epochs) -> int:
    numberOfPeers = int(_getNumberOfPeers()) + 1  # server included
    while hasConnectionsToBeProcessed():
        socket = _getPeerConnection()
        socket.sendall(
            str({"numberOfPeers": numberOfPeers, "epochs": epochs}).encode())
        _finishConnectingClient(socket)
    print("World size sent to all peers")
    _closeTCPConnection()
    return numberOfPeers


def _getNumberOfPeers() -> int:
    global _peerConnections
    return _peerConnections.qsize()


def _getPeerConnection() -> socket:
    global _peerConnections
    peerConnection = _peerConnections.get()
    return peerConnection['socket']


def _closeTCPConnection() -> None:
    global _socketConnection
    if(_socketConnection is not None):
        _stopListeningForConnections()


def _stopListeningForConnections() -> None:
    global _socketConnection
    _socketConnection.close()
    _socketConnection = None
