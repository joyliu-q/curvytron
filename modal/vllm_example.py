# ---
# runtimes: ["gvisor"]
# clouds: ["aws", "oci", "auto"]
# deploy: true
# relative-frequency: 3
# ---
import subprocess
import time

import requests

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install("requests")
)

app = modal.App("vllm-coldstart", image=image)

hf_secret = modal.Secret.from_name("huggingface-secret")


@app.cls(
    gpu="H100",
    secrets=[hf_secret],
    timeout=60 * 60,
)
class VLLM:
    @modal.enter()
    def load(self):
        # Start the VLLM server in the background
        self.process = subprocess.Popen(
            [
                "vllm",
                "serve",
                "--gpu-memory-utilization",
                "0.2",
                "--max-model-len",
                "2048",
                "--max-num-seqs",
                "4",
                "Qwen/Qwen2.5-0.5B-Instruct",
            ],
        )

        # Wait for the server to start up
        self.wait_for_server()

    @modal.web_server(8000)
    def fn(self):
        pass

    def wait_for_server(self):
        """Wait for the VLLM server to be ready"""

        while True:
            try:
                if self.healthcheck():
                    print("VLLM server is ready!")
                    return
            except Exception:
                pass
            time.sleep(1)

    def healthcheck(self):
        """
        Perform a healthcheck on the VLLM server
        Returns True if healthy, False otherwise
        """
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    @modal.exit()
    def cleanup(self):
        """Clean up the background process on exit"""
        if hasattr(self, "process") and self.process:
            self.process.terminate()
            self.process.wait()
