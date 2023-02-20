from sweets import core

if __name__ == "__main__":
    bbox = [-120, 34.5, -118.5, 35.5]
    w = core.Workflow(
        bbox=bbox, start="2018-02-09", end="2018-02-21", track=64, n_workers=2
    )
    # downloaded_files, dem_file, burst_db_file = w.run()
    print(w.run())
