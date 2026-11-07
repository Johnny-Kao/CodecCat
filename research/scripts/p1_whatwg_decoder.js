"use strict";
// Offline research helper. Decode original bytes using Node's ICU-backed
// WHATWG TextDecoder, not the Python codec table. Never output original text.
const crypto = require("node:crypto");
const chunks = [];
process.stdin.on("data", (chunk) => chunks.push(chunk));
process.stdin.on("end", () => {
  const batch = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  const out = batch.map((item) => {
    const raw = Buffer.from(item.data, "base64");
    const checks = {};
    for (const [key, label] of Object.entries(item.labels)) {
      if (typeof label !== "string" || label.length > 100) {
        checks[key] = {valid: false, error: "invalid-label"};
        continue;
      }
      try {
        // fatal=true forbids silently replacing malformed input. Default BOM
        // handling matches public strip semantics for Unicode signatures.
        const decoder = new TextDecoder(label, {fatal: true});
        const text = decoder.decode(raw);
        checks[key] = {
          valid: true,
          encoding: decoder.encoding,
          sha256: crypto.createHash("sha256").update(text, "utf8").digest("hex")
        };
      } catch (error) {
        checks[key] = {valid: false, error: error.name};
      }
    }
    return {id: item.id, checks};
  });
  process.stdout.write(JSON.stringify(out));
});
