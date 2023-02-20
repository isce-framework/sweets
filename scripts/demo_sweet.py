# from shapely import Point

from sweets import core

if __name__ == "__main__":
    bbox = [-120, 34.5, -118.5, 35.5]  # random one
    start, end, track, n_workers, tpw = "2018-02-09", "2018-02-21", 64, 2, 4

    # # midland eq
    # # TexNetEvent(event_id='texnet2022yplg', dt=datetime.datetime(2022, 12, 16, 23, ),
    # # magnitude=5.22863731, latitude=32.19085693, longitude=-102.1406965, depth=8.1923)
    # lon, lat = -102.6406965, 31.6908
    # bbox = Point(lon, lat).buffer(0.5).bounds
    # start, end, track, n_workers, tpw = "2022-10-15", "2023-02-20", 78, 8, 8

    w = core.Workflow(
        bbox=bbox,
        start=start,
        end=end,
        track=track,
        n_workers=n_workers,
        threads_per_worker=tpw,
        log_file="sweets.log",
    )
    # downloaded_files, dem_file, burst_db_file = w.run()
    print(w.run())
