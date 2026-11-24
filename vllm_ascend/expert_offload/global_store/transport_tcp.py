"""Bounded TCP reference transport, explicitly selected for functional tests."""

import json
import secrets
import socket
import socketserver
import struct
import threading

MAX_REQUEST_BYTES = 4096
LENGTH = struct.Struct("!I")


def receive_into(sock, target):
    view = memoryview(target)
    while view:
        count = sock.recv_into(view)
        if not count:
            raise ConnectionError("Expert source disconnected")
        view = view[count:]


class TCPServer:
    def __init__(self, region, address, timeout, max_chunk):
        self.token = secrets.token_hex(32)
        token = self.token

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(timeout)
                header = bytearray(LENGTH.size)
                receive_into(self.request, header)
                length, = LENGTH.unpack(header)
                if length > MAX_REQUEST_BYTES:
                    return
                data = bytearray(length)
                receive_into(self.request, data)
                req = json.loads(data)
                if not secrets.compare_digest(str(req["token"]), token):
                    return
                offset, nbytes = int(req["offset"]), int(req["nbytes"])
                if offset < 0 or nbytes <= 0 or nbytes > max_chunk or offset + nbytes > region.size:
                    return
                self.request.sendall(memoryview(region.mapping)[offset:offset + nbytes])

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = False
            block_on_close = True

        self.server = Server((address, 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True,
                                       name="expert-source-tcp")
        self.thread.start()

    @property
    def port(self):
        return self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class TCPReader:
    def __init__(self, timeout):
        self.timeout = timeout

    def read_into(self, endpoint, offset, destination, nbytes):
        req = json.dumps({"token": endpoint["token"], "offset": offset,
                          "nbytes": nbytes}).encode()
        with socket.create_connection((endpoint["address"], endpoint["port"]), self.timeout) as sock:
            sock.settimeout(self.timeout)
            sock.sendall(LENGTH.pack(len(req)) + req)
            receive_into(sock, destination[:nbytes])
