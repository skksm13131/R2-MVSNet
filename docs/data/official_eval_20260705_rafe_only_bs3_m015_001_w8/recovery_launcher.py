import os
import subprocess
import time

import launcher as base


def matlab_eval(scan_id, ply_path):
    metric_path = os.path.join(base.METRIC_DIR, f"scan{scan_id:03d}.csv")
    matlab_log = os.path.join(base.METRIC_DIR, f"scan{scan_id:03d}.matlab.log")
    matlab_expr = (
        "try, "
        f"addpath('{base.CODE_DIR}'); "
        f"eval_one_scan_dynamic({scan_id}, '{ply_path}', '{metric_path}'); "
        "catch ME, disp(getReport(ME, 'extended')); exit(1); end; exit(0);"
    )
    base.log(f"MATLAB_START scan {scan_id}")
    with open(matlab_log, "w", encoding="utf-8", errors="replace") as output:
        proc = subprocess.Popen(
            [
                base.MATLAB,
                "-nodisplay",
                "-nosplash",
                "-nodesktop",
                "-r",
                matlab_expr,
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        while proc.poll() is None:
            if os.path.isfile(metric_path):
                time.sleep(2)
                base.log(f"MATLAB_METRIC_READY scan {scan_id}; stopping hung MATLAB pid={proc.pid}")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(2)

    base.log(f"MATLAB_EXIT scan {scan_id} code={proc.returncode}")
    if not os.path.isfile(metric_path):
        raise RuntimeError(f"MATLAB failed for scan {scan_id}, see {matlab_log}")
    return metric_path


def evaluate_scan(scan_id):
    metric_path = os.path.join(base.METRIC_DIR, f"scan{scan_id:03d}.csv")
    if os.path.isfile(metric_path):
        base.log(f"RECOVER_EXISTING scan {scan_id}")
        return base.read_metric(metric_path)

    ply = base.pull_ply(scan_id)
    try:
        metric_path = matlab_eval(scan_id, ply)
        return base.read_metric(metric_path)
    finally:
        try:
            os.remove(ply)
            base.log(f"DELETED_LOCAL_PLY scan {scan_id}")
        except FileNotFoundError:
            pass


base.matlab_eval = matlab_eval
base.evaluate_scan = evaluate_scan

if __name__ == "__main__":
    base.main()
