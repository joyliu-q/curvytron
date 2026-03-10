"""
Curvytron LLM Player — deploys a Qwen model via vLLM on Modal
for real-time game move inference.
"""

import threading

import modal
import modal.experimental


# =============================================================================
# Modal App Setup
# =============================================================================

app = modal.App("curvytron-player")

FLASH_PORT = 8000
VLLM_PORT = 8001
MINUTES = 60

MODEL_DIR = "Qwen"
MODEL_NAME = "Qwen3-4B-Instruct-2507"

TARGET_INPUTS = 8
MIN_CONTAINERS = 3

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)



# =============================================================================

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
        "aiohttp>=3.9.0",
        "pydantic>=2.0.0",
        "fastapi[standard]>=0.115.0",
        "uvicorn>=0.30.0",
        "nltk>=3.8.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)




# =============================================================================
# CurvytronPlayer — vLLM inference container
# =============================================================================


@app.cls(
    image=image,
    gpu="H100",
    min_containers=3,
    scaledown_window=15 * MINUTES,
    startup_timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    experimental_options={"flash": "us-east"},
    region="us-east",
)
class CurvytronPlayer:
    @modal.enter()
    def setup(self):
        import subprocess

        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import StreamingResponse
        import aiohttp

        # Start vLLM on VLLM_PORT (internal)
        cmd = [
            "vllm",
            "serve",
            "--uvicorn-log-level=info",
            f"{MODEL_DIR}/{MODEL_NAME}",
            "--served-model-name",
            MODEL_NAME,
            "--port",
            str(VLLM_PORT),
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "8192",
            "--enable-prefix-caching",
            "--guided-decoding-backend",
            "outlines",
            "--override-generation-config",
            '{"enable_thinking": false}',
        ]
        print(" ".join(cmd))
        self._vllm_process = subprocess.Popen(" ".join(cmd), shell=True)

        # Wait for vLLM to be ready
        self._wait_for_port(VLLM_PORT, timeout=600)
        print(f"vLLM ready on port {VLLM_PORT}")

        # Proxy FastAPI app to forward requests to vLLM
        proxy_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @proxy_app.api_route("/{path:path}", methods=["GET", "POST"])
        async def proxy(request: Request, path: str):
            url = f"http://localhost:{VLLM_PORT}/{path}"
            body = await request.body()
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=request.method,
                    url=url,
                    headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                    data=body if body else None,
                ) as resp:
                    content = await resp.read()
                    return StreamingResponse(
                        iter([content]),
                        status_code=resp.status,
                        headers=dict(resp.headers),
                    )

        config = uvicorn.Config(proxy_app, host="0.0.0.0", port=FLASH_PORT)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        self._wait_for_port(FLASH_PORT, timeout=30)
        self.flash_manager = modal.experimental.flash_forward(FLASH_PORT)
        print(f"Flash endpoint ready on port {FLASH_PORT}")

    def _wait_for_port(self, port: int, timeout: int = 30):
        import socket
        import time

        for _ in range(timeout):
            try:
                socket.create_connection(("localhost", port), timeout=1).close()
                return
            except OSError:
                time.sleep(1)
        raise RuntimeError(f"Server failed to start on port {port}")

    @modal.method()
    def keepalive(self):
        pass

    @modal.exit()
    def cleanup(self):
        if hasattr(self, "flash_manager"):
            self.flash_manager.stop()
            self.flash_manager.close()
        if hasattr(self, "_server"):
            self._server.should_exit = True
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5)
        if hasattr(self, "_vllm_process"):
            self._vllm_process.terminate()
            self._vllm_process.wait(timeout=10)

