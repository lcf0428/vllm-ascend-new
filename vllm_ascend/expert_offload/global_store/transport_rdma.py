"""Optional libibverbs backend. No silent TCP fallback for expert bytes."""

import ctypes
import json
import secrets
import socket
import socketserver
import threading

from .transport_tcp import LENGTH, MAX_REQUEST_BYTES, receive_into


class Endpoint(ctypes.Structure):
    _fields_ = [("address", ctypes.c_uint64), ("size", ctypes.c_uint64),
                ("rkey", ctypes.c_uint32), ("qpn", ctypes.c_uint32),
                ("psn", ctypes.c_uint32), ("lid", ctypes.c_uint16),
                ("mtu", ctypes.c_uint16), ("gid", ctypes.c_uint8 * 16)]

    def serialize(self):
        return bytes(self).hex()

    @classmethod
    def deserialize(cls, value):
        raw = bytes.fromhex(value)
        if len(raw) != ctypes.sizeof(cls):
            raise ValueError("RDMA endpoint ABI mismatch")
        return cls.from_buffer_copy(raw)


def send_json(sock, value):
    data = json.dumps(value).encode()
    sock.sendall(LENGTH.pack(len(data)) + data)


def receive_json(sock):
    header = bytearray(LENGTH.size)
    receive_into(sock, header)
    length, = LENGTH.unpack(header)
    if length > MAX_REQUEST_BYTES:
        raise ValueError("RDMA control message too large")
    data = bytearray(length)
    receive_into(sock, data)
    return json.loads(data)


class Verbs:
    def __init__(self, config):
        path = config["rdma_library"]
        if not path:
            raise ValueError("RDMA requires transport.rdma_library; build global_store/rdma_verbs.c")
        self.lib = ctypes.CDLL(path)
        self.config = config
        self.lib.expert_rdma_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                             ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(Endpoint)]
        self.lib.expert_rdma_open.restype = ctypes.c_void_p
        self.lib.expert_rdma_close.argtypes = [ctypes.c_void_p]
        self.lib.expert_rdma_close.restype = None
        self.lib.expert_rdma_connect.argtypes = [ctypes.c_void_p, ctypes.POINTER(Endpoint),
                                                ctypes.POINTER(Endpoint)]
        if not hasattr(self.lib, "expert_rdma_read_into"):
            raise ValueError("RDMA library lacks staging-offset support; rebuild rdma_verbs.c")
        self.lib.expert_rdma_read_into.argtypes = [ctypes.c_void_p, ctypes.POINTER(Endpoint),
                                                  ctypes.c_uint64, ctypes.c_uint64,
                                                  ctypes.c_uint32, ctypes.c_double]
        self.lib.expert_rdma_read_into.restype = ctypes.c_int

    def open(self, address, size):
        endpoint = Endpoint()
        handle = self.lib.expert_rdma_open(self.config["rdma_device"].encode(),
                                           self.config["rdma_port"], self.config["rdma_gid_index"],
                                           address, size, ctypes.byref(endpoint))
        if not handle:
            raise RuntimeError("Cannot open/register Host RDMA region; check device, port, GID and memlock")
        return handle, endpoint

    def connect(self, handle, peer, local):
        if self.lib.expert_rdma_connect(handle, ctypes.byref(peer), ctypes.byref(local)):
            raise RuntimeError("Cannot connect expert RDMA QP")


class RDMAServer:
    def __init__(self, region, config):
        self.verbs = Verbs(config)
        self.token = secrets.token_hex(32)
        token = self.token
        verbs = self.verbs
        self.connections = set()
        self.lock = threading.Lock()
        connections, lock = self.connections, self.lock
        # Fail at startup, including a one-host deployment with rdma selected.
        probe, _ = verbs.open(region.address, region.size)
        verbs.lib.expert_rdma_close(probe)

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                handle = None
                with lock:
                    connections.add(self.request)
                try:
                    self.request.settimeout(config["timeout_s"])
                    req = receive_json(self.request)
                    if not secrets.compare_digest(str(req["token"]), token):
                        return
                    peer = Endpoint.deserialize(req["endpoint"])
                    handle, local = verbs.open(region.address, region.size)
                    verbs.connect(handle, peer, local)
                    send_json(self.request, {"endpoint": local.serialize()})
                    # Keep the MR alive until the client's QP is destroyed.
                    self.request.settimeout(None)
                    self.request.recv(1)
                finally:
                    if handle:
                        verbs.lib.expert_rdma_close(handle)
                    with lock:
                        connections.discard(self.request)

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = False
            block_on_close = True

        self.server = Server((config["bind_address"], 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True,
                                       name="expert-source-rdma")
        self.thread.start()

    @property
    def port(self):
        return self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        with self.lock:
            for sock in self.connections:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        self.server.server_close()
        self.thread.join()


class RDMAReader:
    def __init__(self, config, staging):
        self.verbs = Verbs(config)
        self.timeout = config["timeout_s"]
        self.staging = staging
        self.peers = {}

    def connect(self, rank, endpoint):
        sock = socket.create_connection((endpoint["address"], endpoint["port"]), self.timeout)
        sock.settimeout(self.timeout)
        handle = None
        try:
            handle, local = self.verbs.open(self.staging.data_ptr(), self.staging.numel())
            send_json(sock, {"token": endpoint["token"], "endpoint": local.serialize()})
            peer = Endpoint.deserialize(receive_json(sock)["endpoint"])
            self.verbs.connect(handle, peer, local)
            self.peers[rank] = (handle, peer, sock)
        except BaseException:
            if handle:
                self.verbs.lib.expert_rdma_close(handle)
            sock.close()
            raise

    def read_into(self, rank, endpoint, offset, destination, nbytes):
        if rank not in self.peers:
            self.connect(rank, endpoint)
        handle, peer, _ = self.peers[rank]
        if nbytes <= 0 or nbytes >= (1 << 32) or nbytes > len(destination):
            raise ValueError("Invalid RDMA staging read size")
        address = ctypes.addressof(ctypes.c_ubyte.from_buffer(destination))
        local_offset = address - self.staging.data_ptr()
        if local_offset < 0 or local_offset + nbytes > self.staging.numel():
            raise ValueError("RDMA destination is outside registered staging")
        if self.verbs.lib.expert_rdma_read_into(handle, ctypes.byref(peer), offset,
                                               local_offset, nbytes, self.timeout):
            # Preserve registered memory. Worker failure shuts down this QP.
            raise RuntimeError("Expert RDMA READ failed or timed out")

    def close(self):
        for handle, _, sock in self.peers.values():
            self.verbs.lib.expert_rdma_close(handle)
            sock.close()
        self.peers.clear()
