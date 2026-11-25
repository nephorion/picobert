// Runs the self-tests built into index.html without a browser: `node ci/selftest.js`.
// With no `document`, the page script runs runSelfTests() and sets a non-zero exit code on failure.
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/);
if (!script) throw new Error("no <script> block found in index.html");
vm.runInThisContext(script[1], { filename: "index.html <script>" });
