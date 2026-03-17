from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, Optional

from jupyter_client import BlockingKernelClient
from jupyter_client.connect import write_connection_file


class LocalKernelRuntime:
    """Stateful local IPython kernel runtime bound to a session directory."""

    def __init__(
        self,
        session_dir: str,
        python_executable: str,
        startup_timeout: int = 30,
        execution_timeout: int = 300,
    ) -> None:
        self.session_dir = session_dir
        self.python_executable = python_executable
        self.startup_timeout = startup_timeout
        self.execution_timeout = execution_timeout

        self._client: Optional[BlockingKernelClient] = None
        self._process: Optional[subprocess.Popen] = None
        self._connection_file: Optional[str] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._client is not None and self._process is not None:
            return

        os.makedirs(self.session_dir, exist_ok=True)

        conn_file = os.path.join(self.session_dir, f"kernel-{uuid.uuid4().hex}.json")
        conn_file, _ = write_connection_file(conn_file)
        kernel_cmd = [
            self.python_executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            conn_file,
        ]
        process = subprocess.Popen(
            kernel_cmd,
            cwd=self.session_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            client = BlockingKernelClient(connection_file=conn_file)
            client.load_connection_file()
            client.start_channels()
            client.wait_for_ready(timeout=self.startup_timeout)
        except Exception:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            if os.path.exists(conn_file):
                try:
                    os.remove(conn_file)
                except Exception:
                    pass
            raise

        self._client = client
        self._process = process
        self._connection_file = conn_file

    def execute(self, code: str) -> Dict[str, Any]:
        if self._client is None or self._process is None:
            self.start()

        assert self._client is not None

        with self._lock:
            msg_id = self._client.execute(code)
            deadline = time.time() + self.execution_timeout

            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            result_chunks: list[str] = []
            error_payload: Optional[Dict[str, str]] = None

            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    error_payload = {
                        "name": "TimeoutError",
                        "value": f"Execution timed out after {self.execution_timeout} seconds",
                        "traceback": "",
                    }
                    break

                try:
                    msg = self._client.get_iopub_msg(timeout=remaining)
                except queue.Empty:
                    error_payload = {
                        "name": "TimeoutError",
                        "value": f"Execution timed out after {self.execution_timeout} seconds",
                        "traceback": "",
                    }
                    break

                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                msg_type = msg.get("msg_type")
                content = msg.get("content", {})

                if msg_type == "stream":
                    stream_name = content.get("name")
                    text = content.get("text", "")
                    if stream_name == "stderr":
                        stderr_chunks.append(text)
                    else:
                        stdout_chunks.append(text)
                elif msg_type in {"execute_result", "display_data"}:
                    text_data = content.get("data", {}).get("text/plain")
                    if text_data:
                        result_chunks.append(f"{text_data}\n")
                elif msg_type == "error":
                    error_payload = {
                        "name": content.get("ename", "ExecutionError"),
                        "value": content.get("evalue", ""),
                        "traceback": "\n".join(content.get("traceback", [])),
                    }
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    break

            return {
                "stdout": "".join(stdout_chunks),
                "stderr": "".join(stderr_chunks),
                "result_text": "".join(result_chunks),
                "error": error_payload,
            }

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                self._client.shutdown()
            except Exception:
                pass
            try:
                self._client.stop_channels()
            except Exception:
                pass
            self._client = None

        if self._process is not None:
            try:
                if self._process.poll() is None:
                    self._process.terminate()
                    self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

        if self._connection_file and os.path.exists(self._connection_file):
            try:
                os.remove(self._connection_file)
            except Exception:
                pass
        self._connection_file = None
