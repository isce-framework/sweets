from rich import print

from sweets.core import Workflow

if __name__ == "__main__":
    # start, end, track, n_workers, tpw = "2018-02-09", "2018-02-21", 64, 2, 4

    # midland eq
    # TexNetEvent(event_id='texnet2022yplg', dt=datetime.datetime(2022, 12, 16, 23, ),
    # magnitude=5.22863731, latitude=32.19085693, longitude=-102.1406965, depth=8.1923)
    # lon, lat = -102.1407, 32.1909
    # bbox = Point(lon, lat).buffer(0.05).bounds
    bbox = [-102.2, 32.15, -102.1, 32.22]
    # start, end, track = "2022-10-15", "2023-02-20", 78
    start, end, track = "2022-12-15", "2022-12-29", 78
    n_workers, tpw = 1, 16

    w = Workflow(
        asf_query=dict(
            start=start,
            end=end,
            relativeOrbit=track,
        ),
        bbox=bbox,
        n_workers=n_workers,
        threads_per_worker=tpw,
        max_bandwidth=1,
    )
    print(w.run())  # Print out the final output results
