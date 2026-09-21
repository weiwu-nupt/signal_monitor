import struct
import unittest

import numpy as np

from receiver.rx_protocol import (
    DataProtocolParser,
    SpectrumData,
    SpectrumStitcher,
    Tag,
)


class ProtocolVersionTests(unittest.TestCase):
    def test_version_two_uses_twenty_byte_metadata(self):
        levels = np.array([-120, -90, -70], dtype="<i2")
        payload = struct.pack("<qqiI", 2_000_000_000, 320_000_000, 0, 3)
        payload += levels.tobytes()
        size = 20 + len(payload)
        frame = struct.pack("<II", 2, 1)
        frame += struct.pack("<IIIII", Tag.SPECTRUM, 0, 123, 456, size)
        frame += payload

        parsed = DataProtocolParser().parse_datagram(frame)

        self.assertEqual(len(parsed), 1)
        spectrum = parsed[0]
        self.assertEqual(spectrum.frequency_hz, 2_000_000_000)
        self.assertEqual(spectrum.span_hz, 320_000_000)
        self.assertEqual(spectrum.count, 3)
        np.testing.assert_allclose(spectrum.levels_db, [-12, -9, -7])


class SpectrumStitcherTests(unittest.TestCase):
    def test_accepts_complete_receiver_frame(self):
        stitcher = SpectrumStitcher()
        stitcher.configure(2_000_000_000, 320_000_000)
        fragment = SpectrumData(
            2_000_000_000,
            320_000_000,
            0,
            3,
            np.array([-10, -8, -9], dtype=np.float32),
        )

        complete = stitcher.add(fragment)
        self.assertIsNotNone(complete)
        self.assertEqual(complete.span_hz, 320_000_000)
        np.testing.assert_allclose(complete.levels_db, fragment.levels_db)

    def test_narrow_fragment_still_waits_for_other_slices(self):
        stitcher = SpectrumStitcher()
        stitcher.configure(2_000_000_000, 320_000_000)
        fragment = SpectrumData(
            2_000_000_000,
            100_000_000,
            0,
            3,
            np.array([-10, -8, -9], dtype=np.float32),
        )

        self.assertIsNone(stitcher.add(fragment))


if __name__ == "__main__":
    unittest.main()
