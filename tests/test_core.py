from sweets.core import Workflow


def test_workflow_construct(tmp_path):
    bbox = [-102.2, 32.15, -102.1, 32.22]
    # start, end, track = "2022-10-15", "2023-02-20", 78
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
    # print(w.run())  # Print out the final output results
    outfile = tmp_path / "config.yaml"
    w.to_yaml(outfile, with_comments=True)
    w2 = Workflow.from_yaml(outfile)
    assert w == w2
