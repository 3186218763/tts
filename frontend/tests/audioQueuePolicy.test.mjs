import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import ts from "typescript";

const source = readFileSync(
  new URL("../src/hooks/audioQueuePolicy.ts", import.meta.url),
  "utf8",
);
const output = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const { selectAutoPlayItem } = await import(
  `data:text/javascript;base64,${Buffer.from(output).toString("base64")}`,
);

test("auto-play chooses the newest clip after an earlier delivery completed", () => {
  const first = { key: "first:0", url: "data:audio/wav;base64,first" };
  const recent = { key: "recent:0", url: "data:audio/wav;base64,recent" };

  assert.deepEqual(selectAutoPlayItem([first, recent], null, true), recent);
});
