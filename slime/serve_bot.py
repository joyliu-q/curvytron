"""Serve a curvytron bot via SGLang on Modal.

Hosts SGLang with the trained Qwen3-4B checkpoint.
Clients should use the /generate endpoint with regex constraint
to get clean action output without thinking tokens.

Usage:
    modal run slime/serve_bot.py        # test it
    modal deploy slime/serve_bot.py     # deploy it
"""

from pathlib import Path
import subprocess
import time

import modal
import modal.experimental

MINUTES = 60

MODEL_NAME = "Qwen/Qwen3-4B"
GPU = "H100:1"
PORT = 8000

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("curvytron-checkpoints", create_if_missing=True)

HF_CACHE_PATH = "/root/.cache/huggingface"
CHECKPOINTS_PATH = Path("/checkpoints")
MODEL_PATH = CHECKPOINTS_PATH / "curvytron-selfplay-Qwen3-4B-20260329-040710" / "iter_0000099_hf"

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.9-cu129-amd64-runtime")
    .entrypoint([])
    .pip_install("huggingface-hub==0.36.0")
)

app = modal.App(name="curvytron-bot")


@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: hf_cache_vol, CHECKPOINTS_PATH.as_posix(): checkpoints_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=15 * MINUTES,
    min_containers=1,
)
@modal.experimental.http_server(port=PORT, proxy_regions=["us-east"])
@modal.concurrent(target_inputs=10)
class CurvytronBot:
    @modal.enter()
    def startup(self):
        self.process = _start_server()
        _wait_ready(self.process)
        print("SGLang server ready")

    @modal.exit()
    def stop(self):
        self.process.terminate()
        self.process.wait()


def _start_server() -> subprocess.Popen:
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", str(MODEL_PATH),
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--tp", "1",
        "--mem-fraction-static", "0.85",
        "--context-length", "8192",
    ]
    print("Starting SGLang server:", " ".join(cmd))
    return subprocess.Popen(cmd, start_new_session=True)


def _wait_ready(process: subprocess.Popen, timeout: int = 10 * MINUTES):
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited with code {process.returncode}")
        try:
            requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
            time.sleep(5)
    raise TimeoutError("SGLang server not ready in time")


@app.local_entrypoint()
async def main():
    import aiohttp

    url = (await CurvytronBot._experimental_get_flash_urls.aio())[0]

    # Test using /generate with regex constraint — this is what game clients should use
    system = "You are playing a Snake/Tron game. Respond with ONLY one word: left, straight, or right"
    user = "You are heading toward a wall. Choose: left, straight, or right?"
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

    payload = {
        "text": prompt,
        "sampling_params": {"temperature": 0.1, "max_new_tokens": 5},
        "regex": "(left|straight|right)",
    }

    print(f"Testing bot at {url}/generate")
    headers = {"Modal-Session-Id": "test"}
    async with aiohttp.ClientSession(base_url=url, headers=headers) as session:
        deadline = time.time() + 5 * MINUTES
        while time.time() < deadline:
            try:
                async with session.post("/generate", json=payload, timeout=120) as resp:
                    if resp.status == 503:
                        await __import__("asyncio").sleep(1)
                        continue
                    resp.raise_for_status()
                    result = await resp.json()
                    action = result.get("text", "???")
                    print(f"Bot action: {action}")
                    return
            except Exception as e:
                if "503" in str(e):
                    await __import__("asyncio").sleep(1)
                    continue
                raise
