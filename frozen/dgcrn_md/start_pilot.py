"""Launch a detached pilot with a unique log and a kernel-owned process lock."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / '.pilot_run.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('ANOTHER_PILOT_IS_RUNNING: use the existing log; do not start again')
            return 2
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log = ROOT / ('pilot_console_' + stamp + '.log')
    with log.open('xb') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-B', str(ROOT / 'run_pilot.py'), '--device', 'cuda'],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    cwd=ROOT, start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    (ROOT / 'latest_launch.json').write_text(json.dumps({'pid': process.pid, 'log': str(log),
          'start_request_only_not_completion': True}, indent=2) + '\n', encoding='utf-8')
    print('START_REQUESTED PID=' + str(process.pid))
    print('LOG=' + str(log))
    print('tail -f ' + str(log))
    print('Wait for JOB_START in the log to confirm that checks passed and training actually began.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
