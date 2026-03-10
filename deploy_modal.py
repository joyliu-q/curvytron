import modal
import subprocess
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

subprocess.run(
    ["tar", "czf", "/tmp/curvytron.tar.gz",
     "--exclude=node_modules", "--exclude=bower_components",
     "--exclude=.git", "--exclude=.idea", "--exclude=stats",
     "--exclude=package-lock.json", "-C", ".", "."],
    check=True,
)

app = modal.App.lookup("curvytron", create_if_missing=True)

image = (
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
)

with modal.enable_output():
    sb = modal.Sandbox.create(
        "node",
        "bin/curvytron.js",
        image=image,
        workdir="/app",
        encrypted_ports=[8080],
        timeout=24 * 60 * 60,
        app=app,
    )

tunnel = sb.tunnels()[8080]
print(f"\nCurvytron is live at: {tunnel.url}")
print(f"Sandbox ID: {sb.object_id}")
print("The sandbox will run for up to 24 hours. To stop it early:")
print(f"  modal sandbox terminate {sb.object_id}\n")
