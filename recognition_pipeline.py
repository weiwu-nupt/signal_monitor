"""Shared file/FPGA packet recognition followed by atomic result persistence."""
class PacketRecognitionPipeline:
    def __init__(self, recognizer, store, session_id):
        self.recognizer, self.store, self.session_id = recognizer, store, session_id

    def recognize_iq(self, iq, sample_rate_hz, metadata, **kwargs):
        result = self.recognizer.infer_packet_iq(iq, sample_rate_hz, **kwargs)
        packet_id = self.store.save(self.session_id, result, metadata)
        return result, packet_id

    def recognize_fpga_packet(self, packet, metadata, **kwargs):
        """Accept GPU-resident 32-byte header + 4 MiB IQ; require RF frame context."""
        header, iq = self.recognizer.fpga_iq_packet_from_gpu(packet)
        meta = {**metadata, 'channel_slot': header.sequence,
                'baseband_center_hz': header.center_hz,
                'bandwidth_hz': header.bandwidth_hz}
        # Absolute RF center and narrow/wide type come from the task/transport,
        # which are not both encoded in the fixed IQ header.
        if 'center_hz' not in meta or 'channel_type' not in meta:
            raise ValueError('FPGA persistence requires absolute center_hz and channel_type')
        return self.recognize_iq(iq, header.sample_rate_hz, meta, **kwargs)
