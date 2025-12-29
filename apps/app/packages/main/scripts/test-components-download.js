/*
  Simple script to test /components/download and websocket progress.

  Usage:
    node scripts/test-components-download.js --url http://127.0.0.1:8765 \
      --paths https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_logo.png,https://raw.githubusercontent.com/github/gitignore/main/Node.gitignore

  Notes:
    - Ensure the backend is running.
    - Paths can be any supported sources (http urls, hf repo paths, gs://, s3://, etc.).
*/

const WebSocket = require("ws");

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = argv[i + 1];
    if (a === "--url" && next) {
      args.url = next;
      i++;
      continue;
    }
    if (a === "--paths" && next) {
      args.paths = next;
      i++;
      continue;
    }
  }
  return args;
}

async function main() {
  const { url = "http://127.0.0.1:8765", paths } = parseArgs(process.argv);
  const list = (
    paths
      ? paths.split(",")
      : [
          "https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00001-of-000001.safetensors",
        ]
  )
    .map((s) => s.trim())
    .filter(Boolean);

  console.log("Backend:", url);
  console.log("Paths:", list);

  const res = await fetch(`${url}/components/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paths: list }),
  });
  if (!res.ok) {
    const err = await res.text().catch(() => "");
    throw new Error(`Failed to start: ${res.status} ${res.statusText} ${err}`);
  }
  const data = await res.json();
  const jobId = data.job_id;
  console.log("Started job:", jobId);

  const wsUrl = url.replace("http://", "ws://").replace("https://", "wss://");
  const ws = new WebSocket(`${wsUrl}/ws/job/${jobId}`);

  ws.on("open", () => console.log("WS connected"));
  ws.on("message", (buf) => {
    try {
      const msg = JSON.parse(String(buf));
      const pct =
        typeof msg.progress === "number"
          ? (msg.progress * 100).toFixed(1)
          : "n/a";
      const label = msg?.metadata?.label ? ` [${msg.metadata.label}]` : "";
      console.log(
        `progress=${pct}% status=${msg.status || "processing"}${label} ${msg.message || ""}`,
      );
    } catch (e) {
      console.log("WS message:", String(buf));
    }
  });
  ws.on("error", (e) => console.error("WS error:", e?.message || e));

  // Poll status until completion as a fallback
  let done = false;
  while (!done) {
    await new Promise((r) => setTimeout(r, 1000));
    const s = await fetch(`${url}/components/status/${jobId}`)
      .then((r) => r.json())
      .catch(() => ({}));
    if (
      s &&
      (s.status === "complete" ||
        s.status === "error" ||
        s.status === "cancelled")
    ) {
      console.log("Final status:", s.status);
      done = true;
      try {
        ws.close();
      } catch {}
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
