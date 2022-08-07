# Example of Distributed training using PyTorch

## Configs
copy `.env.example` -> `.env`
modify the params as needed
```bash
SERVER_HOST=0.0.0.0 # the address of the server as per required by the socket library, don't edit if you don't know what you are doing
SERVER_PORT=5001 # the port on which the socket server is running
MASTER_ADDR=... # edit this field as per the master node where the rest of the nodes wiill sinchronize with
MASTER_PORT=29500 # this is the default port, no need to edit

```

### Server
run:
```bash
python server.py
```
A message will ask if all the peers have been connected. Once all peers are connected, type **y** and the training will begin.

### Client
run:
```bash
python client.py
```
The client will receive a rank based on the connection order. Once every peer has been connected, the client will receive the *world_size* and *epochs* as per configured in the master node.