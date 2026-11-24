"""A shared namespace is proven by opening a nonce file, not by hostname."""

import secrets
import socket
from pathlib import Path


def node_identity():
    # boot_id plus a file visibility probe distinguishes separate hosts and
    # separate mount namespaces on one host.
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def create_probe(directory, rank):
    value = secrets.token_hex(32)
    path = directory / f"visibility-{rank}"
    path.write_text(value)
    path.chmod(0o600)
    return {"node": node_identity(), "probe": str(path), "nonce": value}


def can_map(manifest):
    if manifest["node"] != node_identity():
        return False
    try:
        return Path(manifest["probe"]).read_text() == manifest["nonce"]
    except OSError:
        return False


def advertise_address(config):
    return config["advertise_address"] or socket.gethostbyname(socket.gethostname())
