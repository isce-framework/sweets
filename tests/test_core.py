from pathlib import Path
from typing import List

import pytest
from shapely import wkt

from sweets.core import Workflow
from sweets.download import ASFQuery


class TestWorkflow:
    @pytest.fixture(scope="class")
    def bbox(self) -> List[float]:
        return [-102.2, 32.15, -102.1, 32.22]

    def test_workflow_construct(self, tmp_path, bbox):
        start, end, track = "2022-12-15", "2022-12-29", 78
        n_workers, tpw = 1, 16

        w = Workflow(
            asf_query=dict(
                start=start,
                end=end,
                relativeOrbit=track,
                out_dir="data",
            ),
            bbox=bbox,
            n_workers=n_workers,
            threads_per_worker=tpw,
            max_bandwidth=1,
            orbit_dir="orbits",
        )
        outfile = tmp_path / "config.yaml"
        w.to_yaml(outfile, with_comments=True)
        w2 = Workflow.from_yaml(outfile)
        assert w == w2

    def test_workflow_construct_model(self, bbox):
        start, end, track = "2022-12-15", "2022-12-29", 78
        w = Workflow(
            asf_query=ASFQuery(
                start=start, end=end, relativeOrbit=track, out_dir="data", bbox=bbox
            ),
        )
        assert w.bbox == tuple(bbox)

    def test_workflow_bbox_wkt(self, tmp_path):
        start, end, track = "2022-12-15", "2022-12-29", 78
        wkt_str = "POLYGON((-10.0 30.0,-9.0 30.0,-9.0 31.0,-10.0 31.0,-10.0 30.0))"
        loaded_wkt = wkt.loads(wkt_str)

        kwargs = dict(
            asf_query=dict(
                start=start,
                end=end,
                relativeOrbit=track,
            ),
        )
        wkt_bbox = wkt.loads(
            "POLYGON ((-9. 30.0, -9.0 31.0, -10.0 31.0, -10.0 30.0, -9.0 30.0))"
        )
        expected_bbox = (-10, 30, -9, 31)
        w = Workflow(
            bbox=expected_bbox,
            **kwargs,
        )
        assert w.bbox == expected_bbox
        assert _iou(wkt.loads(w.wkt), wkt_bbox) == 1.0

        w = Workflow(
            wkt=wkt_str,
            **kwargs,
        )
        assert w.bbox == expected_bbox
        assert _iou(wkt.loads(w.wkt), loaded_wkt) == 1.0

        wkt_file = tmp_path / "aoi.wkt"
        wkt_file.write_text(wkt_str)
        w = Workflow(
            **kwargs,
            wkt=wkt_file,
        )
        assert _iou(wkt.loads(w.wkt), loaded_wkt) == 1.0
        assert w.bbox == expected_bbox

    def test_workflow_default_factory_order(self, bbox):
        start, end, track = "2022-12-15", "2022-12-29", 78
        dem_path = Path() / "dem"
        mask_path = Path() / "mask"
        w = Workflow(
            water_mask_filename=mask_path,
            dem_filename=dem_path,
            asf_query=ASFQuery(
                start=start, end=end, relativeOrbit=track, out_dir="data", bbox=bbox
            ),
        )
        # assert can set
        assert w.water_mask_filename == mask_path
        assert w.dem_filename == dem_path

        w = Workflow(
            asf_query=ASFQuery(
                start=start, end=end, relativeOrbit=track, out_dir="data", bbox=bbox
            )
        )
        # assert defaults work
        assert w.work_dir / "dem.dat" == w.dem_filename
        assert w.work_dir / "watermask.flg" == w.water_mask_filename

        # assert computed fields work
        assert w.log_dir == w.work_dir / "logs"


def _iou(poly1, poly2):
    return poly1.intersection(poly2).area / poly1.union(poly2).area
