from pathlib import Path
import sys

import numpy as np


class SynchronizedFusionLoader:
    """
    Load synchronized radar and camera data from one RADIal sequence
    using the official RADIal DBReader.
    """

    def __init__(self, sequence_dir, dbreader_dir=None):
        self.sequence_dir = Path(sequence_dir)

        if not self.sequence_dir.exists():
            raise FileNotFoundError(
                f"Sequence not found: {self.sequence_dir}"
            )

        if dbreader_dir is None:
            try:
                from config import get_dbreader_dir
                dbreader_dir = get_dbreader_dir(recording_dir=self.sequence_dir)
            except Exception:
                dbreader_dir = (
                    self.sequence_dir.parents[1]
                    / "RADIal"
                    / "DBReader"
                )

        self.dbreader_dir = Path(dbreader_dir)

        if not self.dbreader_dir.exists():
            raise FileNotFoundError(
                f"DBReader not found: {self.dbreader_dir}"
            )

        sys.path.insert(
            0,
            str(self.dbreader_dir),
        )

        from DBReader.DBReader import SyncReader

        self.reader = SyncReader(
            str(self.sequence_dir),
            master=None,
            silent=False,
        )

    def __len__(self):
        return len(self.reader)

    def load(self, index):
        """
        Load synchronized radar and camera data.

        Returns
        -------
        dict
            {
                "camera": camera frame,
                "radar_ch0": ADC,
                "radar_ch1": ADC,
                "radar_ch2": ADC,
                "radar_ch3": ADC
            }
        """

        if index < 0 or index >= len(self.reader):
            raise IndexError(
                f"Index {index} out of range."
            )

        data = self.reader.GetSensorData(index)

        if "camera" not in data:
            raise RuntimeError(
                "Camera data not available at this index."
            )

        frame = {}

        for channel in (
            "radar_ch0",
            "radar_ch1",
            "radar_ch2",
            "radar_ch3",
        ):
            if channel not in data:
                raise RuntimeError(
                    f"{channel} missing from synchronized frame."
                )

            frame[channel] = np.asarray(
                data[channel]["data"],
                dtype=np.int16,
            )

        frame["camera"] = data["camera"]["data"]

        frame["camera_timestamp"] = (
            data["camera"]["timestamp"]
        )

        frame["radar_timestamp"] = (
            data["radar_ch3"]["timestamp"]
        )

        frame["radar_sample"] = (
            data["radar_ch3"]["sample_number"]
        )

        return frame