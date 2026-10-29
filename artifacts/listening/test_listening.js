// Node check of the page logic (no DOM):  node artifacts/listening/test_listening.js
const assert = require("assert");
const L = require("./listening.js");

const trials = [
  { trial_id: "L01-01", ref: "r1", a: "a1", b: "b1" },
  { trial_id: "L01-02", ref: "r2", a: "a2", b: "b2" },
];
const manifest = { audio_base: "stimuli/", ext: ".wav", files: { r1: {}, a1: {}, b1: {}, r2: {}, a2: {} } };
assert.deepStrictEqual(L.checkTrials(manifest, trials), ["L01-02:b"]);
assert.strictEqual(L.audioUrl(manifest, null, "r1"), "stimuli/r1.wav");
assert.strictEqual(L.audioUrl(manifest, "https://x.org/a", "r1"), "https://x.org/a/r1.wav");

let seen = [];
global.LISTENING_HOOKS = { onResponse: (r) => seen.push(r.trial_id), onFinish: (e) => seen.push("fin" + e.responses.length) };
const s = L.createSession({ set: "pilot", listener: "L01" }, trials);
const ok = { sim_a: 3, sim_b: 4, nat_a: 5, nat_b: 2, pref: "B", listened: { ref: true, a: true, b: true }, plays: { ref: 1, a: 2, b: 1 }, rt_s: 12.3 };
assert.ok(s.record("L01-01", { ...ok, listened: { ref: true, a: true, b: false } }).length > 0, "unlistened clip rejected");
assert.ok(s.record("L01-01", { ...ok, sim_a: 6 }).length > 0, "out of range rejected");
assert.ok(s.record("L01-01", { ...ok, pref: null }).length > 0, "missing preference rejected");
assert.strictEqual(s.count(), 0);
assert.strictEqual(s.next(), 0);
assert.deepStrictEqual(s.record("L01-01", ok), []);
assert.strictEqual(s.next(), 1);
assert.ok(!s.isDone());
assert.throws(() => s.record("nope", ok));
assert.deepStrictEqual(s.record("L01-02", { ...ok, pref: "A" }), []);
assert.ok(s.isDone() && s.finished);
assert.deepStrictEqual(seen, ["L01-01", "L01-02", "fin2"]);
const e = s.export();
assert.strictEqual(e.schema, 1);
assert.strictEqual(e.responses.length, 2);
assert.strictEqual(e.responses[1].pref, "A");
JSON.parse(JSON.stringify(e));
console.log("listening.js logic ok");
