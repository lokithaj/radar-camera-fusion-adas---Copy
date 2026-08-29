from pathlib import Path

from radar.src.radial_loader import RADIalLoader


DATASET = (
    Path.home()
    / "Desktop"
    / "RADIal_data"
    / "RECORD@2020-11-21_13.44.44"
)

DBREADER = (
    Path.home()
    / "Desktop"
    / "RADIal"
    / "DBReader"
)


def test_load_real_radial_frame():
    loader = RADIalLoader(
        DATASET,
        dbreader_dir=DBREADER,
    )

    print(f"\nNumber of synchronized frames: {len(loader)}")

    frame = loader.load_frame(0)

    for channel, data in frame.items():
        print(
            f"{channel}: "
            f"shape={data.shape}, "
            f"dtype={data.dtype}, "
            f"samples={data.size}"
        )

        assert data.dtype == "int16"
        assert data.size > 0

    metadata = loader.load_frame_metadata(0)

    print("\nRadar metadata:")

    for channel, info in metadata.items():
        print(
            f"{channel}: "
            f"sample={info['sample_number']}, "
            f"index={info['index']}"
        )