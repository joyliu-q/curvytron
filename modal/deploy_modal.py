import os
import subprocess
import time

import modal
import modal.experimental

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = modal.App(name="curvytron")

# Image: system deps + node + global tools only (rarely changes)
curvytron_image = (
    modal.Image.debian_slim(python_version="3.10")
    .run_commands(
        "apt-get update && apt-get install -y curl xz-utils git sassc",
        "curl -fsSL https://nodejs.org/dist/v10.24.1/node-v10.24.1-linux-x64.tar.xz | tar -xJ -C /usr/local --strip-components=1",
        "npm install -g bower gulp-cli",
    )
    .pip_install("httpx")
    .add_local_dir(
        PROJECT_DIR,
        remote_path="/app",
        ignore=["node_modules", ".git", "__pycache__", "modal", "uv.lock", ".claude", "scripts", "src"],
        copy=True
    )
    .run_commands(
        # Install deps inside the image so they're cached and don't re-run on every container start.
        # --force handles conflicts with stale .bin symlinks (e.g. sass).
        "cd /app && npm install --ignore-scripts --force && bower install --allow-root -F",
    )
    .add_local_dir(
        f"{PROJECT_DIR}/src",
        remote_path="/app/src",
        copy=True
    )
)


with curvytron_image.imports():
    import httpx


@app.cls(
    image=curvytron_image,
    experimental_options={"flash": "us-east"},
    region="us-east",
    min_containers=1,
)
class Curvytron:
    @modal.enter()
    def start(self):
        subprocess.run(
            "cp -n config.json.sample config.json",
            cwd="/app", shell=True, check=True,
        )
        subprocess.run(
            "mkdir -p /app/web/css && sassc --style compressed /app/src/sass/style.scss /app/web/css/style.css",
            shell=True, check=True,
        )
        # Stub out the sass module to avoid native build issues
        subprocess.run(
            "rm -rf /app/node_modules/sass && mkdir -p /app/node_modules/sass"
            " && echo 'module.exports = {};' > /app/node_modules/sass/index.js"
            ' && echo \'{"name":"sass","main":"index.js"}\' > /app/node_modules/sass/package.json',
            shell=True, check=True,
        )
        subprocess.run(
            "gulp jshint server front-expose ga views front-min",
            cwd="/app", shell=True, check=True,
        )

        self.process = subprocess.Popen(
            ["node", "bin/curvytron.js"],
            cwd="/app",
        )

        # Wait for the server to become healthy
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                with httpx.Client(timeout=2) as client:
                    resp = client.get("http://127.0.0.1:8080/")
                    if resp.status_code == 200:
                        print("Curvytron server is up!")
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Curvytron server failed to start within 60s")

        self.flash_handle = modal.experimental.flash_forward(8080, process=self.process)

    @modal.method()
    def method(self):
        pass

    @modal.exit()
    def stop(self):
        self.flash_handle.stop()
        time.sleep(5)
        self.flash_handle.close()


@app.local_entrypoint()
def main():
    Curvytron().method.remote()
