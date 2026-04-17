import time

from ipc.message_schema import StatusPacket
from ipc.transport.uds_seqpacket import UdsSeqpacketClient


class StatusPublisher:
    def __init__(self, socket_path=None, source=None, connect_timeout=1.0):
        self.source = source
        self.sequence = 0
        self.client = None
        if socket_path is None or source is None:
            return

        try:
            self.client = UdsSeqpacketClient(
                socket_path=socket_path,
                retry_interval=0.05,
                connect_timeout=connect_timeout,
            )
        except TimeoutError:
            self.client = None

    @property
    def enabled(self):
        return self.client is not None

    def publish(self, channel_id=0, payload_dict=None):
        if self.client is None:
            return False

        packet = StatusPacket.from_dict(
            sequence=self.sequence,
            monotonic_ns=time.monotonic_ns(),
            source=self.source,
            channel_id=channel_id,
            payload_dict=payload_dict or {},
        )
        self.sequence += 1
        return self.client.try_send(packet.encode())

    def close(self):
        if self.client is None:
            return
        self.client.close()
        self.client = None