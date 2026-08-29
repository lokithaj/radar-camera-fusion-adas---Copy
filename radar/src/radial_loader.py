"""
RADIal radar data loader.

Loads synchronized radar frames using the official RADIal DBReader.
Only radar data is exposed to the rest of the project.
"""

from pathlib import Path
import sys
import numpy as np


class RADIalLoader:
    """Load synchronized radar frames from a RADIal sequence."""

    RADAR_CHANNELS = (
        "radar_ch0",
        "radar_ch1",
        "radar_ch2",
        "radar_ch3",
    )

    def __init__(self, sequence_dir, dbreader_dir=None):
        self.sequence_dir = Path(sequence_dir)

        if not self.sequence_dir.exists():
            raise FileNotFoundError(
                f"RADIal sequence directory not found: {self.sequence_dir}"
            )

        # Location of the official RADIal DBReader.
        if dbreader_dir is None:
            dbreader_dir = (
                self.sequence_dir.parents[1]
                / "RADIal"
                / "DBReader"
            )

        self.dbreader_dir = Path(dbreader_dir)

        if not self.dbreader_dir.exists():
            raise FileNotFoundError(
                f"RADIal DBReader not found: {self.dbreader_dir}"
            )

        # Import the official RADIal reader.
        sys.path.insert(0, str(self.dbreader_dir))

        from DBReader.DBReader import SyncReader

        self.reader = SyncReader(
            str(self.sequence_dir),
            master=None,
            silent=False,
        )

        self.num_frames = len(self.reader)

    def __len__(self):
        return self.num_frames

    def load_frame(self, index):
        """
        Load one synchronized radar frame.

        Returns
        -------
        dict
            radar_ch0, radar_ch1, radar_ch2, radar_ch3
            containing int16 ADC samples.
        """

        if index < 0 or index >= self.num_frames:
            raise IndexError(
                f"Frame index {index} out of range "
                f"[0, {self.num_frames - 1}]"
            )

        sensor_data = self.reader.GetSensorData(index)

        frame = {}

        for channel in self.RADAR_CHANNELS:
            if channel not in sensor_data:
                raise RuntimeError(
                    f"{channel} was not found in synchronized frame."
                )

            frame[channel] = np.asarray(
                sensor_data[channel]["data"],
                dtype=np.int16,
            )

        return frame

    def load_frame_metadata(self, index):
        """Return metadata for the four synchronized radar channels."""

        if index < 0 or index >= self.num_frames:
            raise IndexError(
                f"Frame index {index} out of range "
                f"[0, {self.num_frames - 1}]"
            )

        sensor_data = self.reader.GetSensorData(index)

        metadata = {}

        for channel in self.RADAR_CHANNELS:
            metadata[channel] = {
                "timestamp": sensor_data[channel]["timestamp"],
                "sample_number": sensor_data[channel]["sample_number"],
                "index": sensor_data[channel]["index"],
                "offset_byte": sensor_data[channel]["offset_byte"],
            }

        return metadata