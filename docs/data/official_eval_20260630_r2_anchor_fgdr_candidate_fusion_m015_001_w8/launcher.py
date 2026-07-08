import concurrent.futures
import csv
import os
import subprocess
import time
from datetime import datetime

import paramiko


TRAIN_HOST = "10.91.28.4"
TRAIN_PORT = 21785
TRAIN_USER = "u104754251515"
TRAIN_PASSWORD = "Ab107129!!"

TRAIN_REPO = "/home/u104754251515/baseline/CasMVSNet20260604"
TAG = "20260630_r2_anchor_fgdr_candidate_fusion_m015_001"
REMOTE_PLY_DIR = f"{TRAIN_REPO}/outputs_retest/{TAG}"
REMOTE_RESULT_DIR = f"{TRAIN_REPO}/results_m/official_matlab_{TAG}_w8"

RUN_DIR = f"/root/official_eval_{TAG}_w8"
PLY_TMP = os.path.join(RUN_DIR, "ply_tmp")
METRIC_DIR = os.path.join(RUN_DIR, "metrics")
CODE_DIR = "/root/official_eval_batch_parallel_20260514/code"
MATLAB = "/opt/app/matlab/R2022a/bin/matlab"
MAX_WORKERS = 8
USED_SETS = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(RUN_DIR, "launcher.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def connect_train():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(TRAIN_HOST, port=TRAIN_PORT, username=TRAIN_USER, password=TRAIN_PASSWORD,
                   timeout=30, banner_timeout=30, auth_timeout=30)
    return client


def sftp_mkdirs(sftp, path):
    current = ""
    for part in path.strip("/").split("/"):
        current += "/" + part
        try:
            sftp.stat(current)
        except IOError:
            sftp.mkdir(current)


def pull_ply(scan_id):
    filename = f"mvsnet{scan_id:03d}_l3.ply"
    remote = f"{REMOTE_PLY_DIR}/{filename}"
    local = os.path.join(PLY_TMP, filename)
    start = time.time()
    client = connect_train()
    try:
        sftp = client.open_sftp()
        size = sftp.stat(remote).st_size
        log(f"PULL {filename} size={size}")
        last_report = {"t": time.time()}

        def callback(done, total):
            now = time.time()
            if done == total or now - last_report["t"] > 20:
                pct = 100.0 * done / max(total, 1)
                log(f"PULL_PROGRESS {filename} {done}/{total} {pct:.1f}%")
                last_report["t"] = now

        sftp.get(remote, local, callback=callback)
        sftp.close()
    finally:
        client.close()
    log(f"PULL_DONE {filename} elapsed={time.time() - start:.1f}s")
    return local


def matlab_eval(scan_id, ply_path):
    metric_path = os.path.join(METRIC_DIR, f"scan{scan_id:03d}.csv")
    matlab_log = os.path.join(METRIC_DIR, f"scan{scan_id:03d}.matlab.log")
    matlab_expr = (
        "try, "
        f"addpath('{CODE_DIR}'); "
        f"eval_one_scan_dynamic({scan_id}, '{ply_path}', '{metric_path}'); "
        "catch ME, disp(getReport(ME, 'extended')); exit(1); end; exit(0);"
    )
    log(f"MATLAB_START scan {scan_id}")
    with open(matlab_log, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run([MATLAB, "-nodisplay", "-nosplash", "-nodesktop", "-r", matlab_expr],
                              stdout=f, stderr=subprocess.STDOUT)
    log(f"MATLAB_EXIT scan {scan_id} code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"MATLAB failed for scan {scan_id}, see {matlab_log}")
    return metric_path


def read_metric(metric_path):
    with open(metric_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = rows[0]
    return int(row["ScanID"]), float(row["Acc_Mean"]), float(row["Comp_Mean"]), float(row["Overall"])


def evaluate_scan(scan_id):
    ply = pull_ply(scan_id)
    try:
        metric = matlab_eval(scan_id, ply)
        result = read_metric(metric)
    finally:
        try:
            os.remove(ply)
            log(f"DELETED_LOCAL_PLY scan {scan_id}")
        except FileNotFoundError:
            pass
    return result


def write_final(results):
    results = sorted(results, key=lambda x: x[0])
    final_csv = os.path.join(RUN_DIR, "official_results.csv")
    avg_acc = sum(r[1] for r in results) / len(results)
    avg_comp = sum(r[2] for r in results) / len(results)
    avg_overall = sum(r[3] for r in results) / len(results)
    with open(final_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ScanID", "Acc_Mean", "Comp_Mean", "Overall"])
        for scan_id, acc, comp, overall in results:
            writer.writerow([scan_id, f"{acc:.6f}", f"{comp:.6f}", f"{overall:.6f}"])
        writer.writerow([])
        writer.writerow(["AVERAGE", f"{avg_acc:.6f}", f"{avg_comp:.6f}", f"{avg_overall:.6f}"])
    summary = os.path.join(RUN_DIR, "SUMMARY.md")
    with open(summary, "w", encoding="utf-8") as f:
        f.write("# Official MATLAB evaluation: 20260630_r2_anchor_fgdr_candidate_fusion_m015_001\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"Point cloud dir: `{REMOTE_PLY_DIR}`  \n")
        f.write(f"Eval machine dir: `{RUN_DIR}`  \n\n")
        f.write("## Official MATLAB Results\n\n")
        f.write("```text\n")
        f.write(f"Overall Acc  Mean: {avg_acc:.6f}\n")
        f.write(f"Overall Comp Mean: {avg_comp:.6f}\n")
        f.write(f"Total Overall    : {avg_overall:.6f}\n")
        f.write("```\n")
    return final_csv, summary, avg_acc, avg_comp, avg_overall


def sync_back(files):
    client = connect_train()
    try:
        sftp = client.open_sftp()
        sftp_mkdirs(sftp, REMOTE_RESULT_DIR)
        for path in files:
            sftp.put(path, f"{REMOTE_RESULT_DIR}/{os.path.basename(path)}")
        remote_metric_dir = f"{REMOTE_RESULT_DIR}/metrics"
        sftp_mkdirs(sftp, remote_metric_dir)
        for name in os.listdir(METRIC_DIR):
            local = os.path.join(METRIC_DIR, name)
            if os.path.isfile(local):
                sftp.put(local, f"{remote_metric_dir}/{name}")
        sftp.close()
    finally:
        client.close()


def main():
    os.makedirs(PLY_TMP, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)
    log(f"START official_matlab_{TAG}_w8 workers={MAX_WORKERS}")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_scan = {executor.submit(evaluate_scan, scan): scan for scan in USED_SETS}
        for future in concurrent.futures.as_completed(future_to_scan):
            scan = future_to_scan[future]
            result = future.result()
            results.append(result)
            avg = sum(r[3] for r in results) / len(results)
            log(f"SCAN_DONE {scan} row={result[0]},{result[1]:.6f},{result[2]:.6f},{result[3]:.6f} "
                f"status=done finished={len(results)}/{len(USED_SETS)} running_average={avg:.6f}")
    final_csv, summary, avg_acc, avg_comp, avg_overall = write_final(results)
    log(f"FINAL acc={avg_acc:.6f} comp={avg_comp:.6f} overall={avg_overall:.6f}")
    log(f"SYNC_BACK_START {REMOTE_RESULT_DIR}")
    sync_back([final_csv, summary, os.path.join(RUN_DIR, "launcher.log")])
    log("SYNC_BACK_DONE")
    log(f"ALL_DONE official_matlab_{TAG}_w8")


if __name__ == "__main__":
    main()
