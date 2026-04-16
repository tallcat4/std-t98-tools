import errno
import os
import select
import socket
import time


DEFAULT_MULTI_FRAME_SOCKET_PATH = os.environ.get(
    "STD_T98_MULTI_FRAME_SOCKET",
    "/tmp/std_t98_multi_frame.sock",
)
DEFAULT_MULTI_VOICE_SOCKET_PATH = os.environ.get(
    "STD_T98_MULTI_VOICE_SOCKET",
    "/tmp/std_t98_multi_voice.sock",
)
DEFAULT_MULTI_STATUS_SOCKET_PATH = os.environ.get(
    "STD_T98_MULTI_STATUS_SOCKET",
    "/tmp/std_t98_multi_status.sock",
)


def resolve_frame_socket_path(channel_count=1, socket_path=None):
    del channel_count
    return socket_path or DEFAULT_MULTI_FRAME_SOCKET_PATH


def resolve_voice_socket_path(channel_count=1, socket_path=None):
    del channel_count
    return socket_path or DEFAULT_MULTI_VOICE_SOCKET_PATH


def resolve_status_socket_path(channel_count=1, socket_path=None):
    del channel_count
    return socket_path or DEFAULT_MULTI_STATUS_SOCKET_PATH


class UdsSeqpacketServer:
    def __init__(self, socket_path, backlog=1):
        self.socket_path = socket_path
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.listener.setblocking(False)
        self.client = None

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self.listener.bind(self.socket_path)
        self.listener.listen(backlog)

    def _accept_client(self):
        if self.client is not None:
            return True

        try:
            client, _ = self.listener.accept()
        except BlockingIOError:
            return False

        client.setblocking(False)
        self.client = client
        return True

    def _drop_client(self):
        if self.client is None:
            return
        try:
            self.client.close()
        finally:
            self.client = None

    def send(self, payload: bytes) -> bool:
        if not self._accept_client():
            return False

        try:
            sent = self.client.send(payload)
            if sent != len(payload):
                raise OSError(errno.EPIPE, "partial packet send")
            return True
        except OSError:
            self._drop_client()
            return False

    def close(self):
        self._drop_client()
        try:
            self.listener.close()
        finally:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)


class UdsSeqpacketClient:
    def __init__(self, socket_path, retry_interval=0.25, connect_timeout=None):
        self.socket_path = socket_path
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._connect(retry_interval=retry_interval, connect_timeout=connect_timeout)

    def _connect(self, retry_interval, connect_timeout):
        deadline = None if connect_timeout is None else time.monotonic() + connect_timeout
        while True:
            try:
                self.socket.connect(self.socket_path)
                return
            except OSError as exc:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out connecting to {self.socket_path}") from exc
                if exc.errno not in (errno.ENOENT, errno.ECONNREFUSED, errno.EAGAIN):
                    raise
                time.sleep(retry_interval)

    def fileno(self):
        return self.socket.fileno()

    def send(self, payload: bytes):
        return self.socket.send(payload)

    def recv(self, max_size=65536):
        return self.socket.recv(max_size)

    def close(self):
        self.socket.close()


class UdsSeqpacketReceiver:
    def __init__(self, socket_path, backlog=8, max_size=65536):
        self.socket_path = socket_path
        self.max_size = max_size
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.listener.setblocking(False)
        self.clients = {}
        self.poller = select.poll()

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self.listener.bind(self.socket_path)
        self.listener.listen(backlog)
        self.poller.register(self.listener.fileno(), select.POLLIN)

    def _accept_pending_clients(self):
        while True:
            try:
                client, _ = self.listener.accept()
            except BlockingIOError:
                return

            client.setblocking(False)
            fileno = client.fileno()
            self.clients[fileno] = client
            self.poller.register(fileno, select.POLLIN | select.POLLHUP | select.POLLERR | select.POLLNVAL)

    def _drop_client(self, fileno):
        client = self.clients.pop(fileno, None)
        if client is None:
            return

        try:
            self.poller.unregister(fileno)
        except KeyError:
            pass
        finally:
            client.close()

    def recv(self, timeout_ms=None):
        poll_timeout = -1 if timeout_ms is None else timeout_ms

        while True:
            events = self.poller.poll(poll_timeout)
            if not events:
                return None

            poll_timeout = 0
            for fileno, mask in events:
                if fileno == self.listener.fileno():
                    self._accept_pending_clients()
                    continue

                if mask & (select.POLLHUP | select.POLLERR | select.POLLNVAL):
                    self._drop_client(fileno)
                    continue

                client = self.clients.get(fileno)
                if client is None:
                    continue

                try:
                    payload = client.recv(self.max_size)
                except OSError:
                    self._drop_client(fileno)
                    continue

                if not payload:
                    self._drop_client(fileno)
                    continue
                return payload

    def close(self):
        for fileno in list(self.clients):
            self._drop_client(fileno)

        try:
            self.poller.unregister(self.listener.fileno())
        except KeyError:
            pass

        try:
            self.listener.close()
        finally:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
