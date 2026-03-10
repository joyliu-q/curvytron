import os
import subprocess
import time

import modal
import modal.experimental

# --- Build the project tarball locally before image construction ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run(
    [
        "tar", "czf", "/tmp/curvytron.tar.gz",
        "--exclude=node_modules", "--exclude=bower_components",
        "--exclude=.git", "--exclude=.idea", "--exclude=stats",
        "--exclude=package-lock.json", "-C", ".", ".",
    ],
    check=True,
)

app = modal.App(name="curvytron")

curvytron_image = (
    modal.Image.debian_slim(python_version="3.10")
    .run_commands(
        "apt-get update && apt-get install -y curl xz-utils git sassc",
        "curl -fsSL https://nodejs.org/dist/v10.24.1/node-v10.24.1-linux-x64.tar.xz | tar -xJ -C /usr/local --strip-components=1",
        "npm install -g bower gulp-cli",
    )
    .add_local_file("/tmp/curvytron.tar.gz", remote_path="/tmp/curvytron.tar.gz", copy=True)
    .run_commands(
        "mkdir -p /app && tar xzf /tmp/curvytron.tar.gz -C /app && rm /tmp/curvytron.tar.gz",
        "cd /app && npm install --ignore-scripts && bower install --allow-root -F",
        "cd /app && cp config.json.sample config.json",
        "mkdir -p /app/web/css && sassc --style compressed /app/src/sass/style.scss /app/web/css/style.css",
        "rm -rf /app/node_modules/sass && mkdir -p /app/node_modules/sass && echo 'module.exports = {};' > /app/node_modules/sass/index.js && echo '{\"name\":\"sass\",\"main\":\"index.js\"}' > /app/node_modules/sass/package.json",
        "cd /app && gulp jshint server front-expose ga views front-min",
    )
    .pip_install("httpx")
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
