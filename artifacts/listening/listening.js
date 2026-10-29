/* Phase 8.3 listening test, page logic. Draft: responses live in a JS object; storage is connected by the host page.
 *
 * Data (static files next to the page, or chosen with the file inputs when opened from disk):
 *   manifest.json  {audio_base, ext, files:{<hash>:{dur_s}}}
 *   <set>.json     {set, seed, listeners:{<id>:[{trial_id, ref, a, b}]}}      (trials; no arm labels)
 * URL: index.html?set=pilot&listener=L01[&manifest=manifest.json][&trials=trials/pilot.json][&audio=<base url>]
 *
 * Storage hook: before use the host may set  window.LISTENING_HOOKS = {onResponse(record, session), onFinish(exported)}.
 * Both are optional and called inside try/catch; the in-memory session is always the source of truth until exported.
 * Export format (session.export()):
 *   {schema:1, set, listener, started, finished, user_agent, responses:[{trial_id, sim_a, sim_b, nat_a, nat_b, pref:"A"|"B",
 *    plays:{ref,a,b}, listened:{ref,a,b}, rt_s, t}]}
 */
(function (root) {
  "use strict";

  var CLIPS = ["ref", "a", "b"];

  /** In-memory session: ordered trials, responses by trial id, validation and export. Pure (no DOM). */
  function createSession(meta, trials) {
    var responses = {};
    var order = trials.map(function (t) { return t.trial_id; });
    var s = {
      meta: meta,
      trials: trials,
      started: new Date().toISOString(),
      finished: null,
      responses: responses,
      count: function () { return Object.keys(responses).length; },
      isDone: function () { return Object.keys(responses).length >= trials.length; },
      next: function () {
        for (var i = 0; i < order.length; i++) { if (!responses[order[i]]) return i; }
        return -1;
      },
      validate: function (r) {
        var errs = [];
        ["sim_a", "sim_b", "nat_a", "nat_b"].forEach(function (k) {
          var v = r[k];
          if (!(Number.isInteger(v) && v >= 1 && v <= 5)) errs.push(k + " must be 1-5");
        });
        if (r.pref !== "A" && r.pref !== "B") errs.push("choose A or B");
        CLIPS.forEach(function (c) { if (!(r.listened && r.listened[c])) errs.push("listen to the whole " + c.toUpperCase() + " clip"); });
        return errs;
      },
      record: function (tid, r) {
        if (order.indexOf(tid) < 0) throw new Error("unknown trial " + tid);
        var errs = s.validate(r);
        if (errs.length) return errs;
        var rec = {
          trial_id: tid, sim_a: r.sim_a, sim_b: r.sim_b, nat_a: r.nat_a, nat_b: r.nat_b, pref: r.pref,
          plays: r.plays || {}, listened: r.listened || {}, rt_s: r.rt_s == null ? null : r.rt_s, t: new Date().toISOString()
        };
        responses[tid] = rec;
        if (s.isDone() && !s.finished) s.finished = new Date().toISOString();
        var hooks = root.LISTENING_HOOKS;
        if (hooks && hooks.onResponse) { try { hooks.onResponse(rec, s); } catch (e) { /* host storage must not break the test */ } }
        if (s.finished && hooks && hooks.onFinish) { try { hooks.onFinish(s.export()); } catch (e) { /* ditto */ } }
        return [];
      },
      export: function () {
        return {
          schema: 1, set: meta.set, listener: meta.listener, started: s.started, finished: s.finished,
          user_agent: (typeof navigator !== "undefined" ? navigator.userAgent : "node"),
          responses: order.filter(function (t) { return responses[t]; }).map(function (t) { return responses[t]; })
        };
      }
    };
    return s;
  }

  function audioUrl(manifest, base, hash) {
    var b = base != null ? base : manifest.audio_base;
    if (b && b.slice(-1) !== "/") b += "/";
    return (b || "") + hash + (manifest.ext || ".wav");
  }

  function checkTrials(manifest, trials) {
    var missing = [];
    trials.forEach(function (t) { ["ref", "a", "b"].forEach(function (k) { if (!manifest.files[t[k]]) missing.push(t.trial_id + ":" + k); }); });
    return missing;
  }

  var api = { createSession: createSession, audioUrl: audioUrl, checkTrials: checkTrials };
  root.Listening = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;

  if (typeof document === "undefined") return;

  // ------------------------------------------------------------------ UI
  var $ = function (id) { return document.getElementById(id); };
  var state = { session: null, manifest: null, base: null, idx: -1, shownAt: 0, plays: {}, listened: {}, startedTrial: null };

  function qs(name) { return new URLSearchParams(location.search).get(name); }
  function setMsg(t, bad) { var m = $("msg"); m.textContent = t || ""; m.className = bad ? "bad" : ""; }

  function fetchJson(url) {
    return fetch(url).then(function (r) { if (!r.ok) throw new Error(url + ": HTTP " + r.status); return r.json(); });
  }
  function readFile(file) {
    return new Promise(function (res, rej) {
      var fr = new FileReader();
      fr.onload = function () { try { res(JSON.parse(fr.result)); } catch (e) { rej(e); } };
      fr.onerror = function () { rej(fr.error); };
      fr.readAsText(file);
    });
  }

  function start(manifest, trialSet) {
    var lid = qs("listener") || $("listener").value.trim();
    var lst = trialSet.listeners[lid];
    if (!lst) { setMsg("Unknown listener id '" + lid + "'. Available: " + Object.keys(trialSet.listeners).join(", "), true); return; }
    var missing = checkTrials(manifest, lst);
    if (missing.length) { setMsg("Manifest lacks files for " + missing.slice(0, 3).join(", ") + (missing.length > 3 ? ", ..." : ""), true); return; }
    state.manifest = manifest;
    state.base = qs("audio");
    state.session = createSession({ set: trialSet.set, listener: lid }, lst);
    $("setup").hidden = true;
    $("intro").hidden = false;
    $("who").textContent = lid;
    $("ntr").textContent = String(lst.length);
  }

  function clipEl(kind) { return $("au_" + kind); }

  function showTrial() {
    var s = state.session;
    state.idx = s.next();
    if (state.idx < 0) { finish(); return; }
    var t = s.trials[state.idx];
    $("trial").hidden = false;
    $("progress").textContent = "Trial " + (state.idx + 1) + " of " + s.trials.length;
    state.plays = { ref: 0, a: 0, b: 0 };
    state.listened = { ref: false, a: false, b: false };
    CLIPS.forEach(function (k) {
      var a = clipEl(k);
      a.pause();
      a.src = audioUrl(state.manifest, state.base, t[k]);
      a.load();
      $("st_" + k).textContent = "not played yet";
    });
    document.querySelectorAll("#trial input[type=radio]").forEach(function (r) { r.checked = false; });
    $("submit").disabled = false;
    setMsg("");
    state.shownAt = Date.now();
    window.scrollTo(0, 0);
  }

  function wireAudio() {
    CLIPS.forEach(function (k) {
      var a = clipEl(k);
      a.addEventListener("play", function () {
        state.plays[k] += 1;
        CLIPS.forEach(function (o) { if (o !== k) clipEl(o).pause(); });
      });
      a.addEventListener("ended", function () { state.listened[k] = true; $("st_" + k).textContent = "listened (" + state.plays[k] + "x)"; });
      a.addEventListener("error", function () { setMsg("Could not load audio for clip " + k.toUpperCase() + ". Check the audio location.", true); });
    });
  }

  function val(name) {
    var r = document.querySelector("input[name=" + name + "]:checked");
    return r ? (name === "pref" ? r.value : parseInt(r.value, 10)) : null;
  }

  function submit() {
    var s = state.session, t = s.trials[state.idx];
    var errs = s.record(t.trial_id, {
      sim_a: val("sim_a"), sim_b: val("sim_b"), nat_a: val("nat_a"), nat_b: val("nat_b"), pref: val("pref"),
      plays: state.plays, listened: state.listened, rt_s: Math.round((Date.now() - state.shownAt) / 100) / 10
    });
    if (errs.length) { setMsg("Please complete: " + errs.join("; "), true); return; }
    showTrial();
  }

  function download() {
    var data = JSON.stringify(state.session.export(), null, 1);
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([data], { type: "application/json" }));
    a.download = "responses_" + state.session.meta.set + "_" + state.session.meta.listener + ".json";
    a.click();
  }

  function finish() {
    $("trial").hidden = true;
    $("done").hidden = false;
    $("exportbox").value = JSON.stringify(state.session.export());
  }

  function buildScales() {
    document.querySelectorAll(".scale").forEach(function (div) {
      var name = div.getAttribute("data-name");
      for (var v = 1; v <= 5; v++) {
        var lab = document.createElement("label");
        var inp = document.createElement("input");
        inp.type = "radio"; inp.name = name; inp.value = String(v);
        lab.appendChild(inp);
        lab.appendChild(document.createTextNode(" " + v));
        div.appendChild(lab);
      }
    });
  }

  function init() {
    buildScales();
    wireAudio();
    var man = qs("manifest") || "manifest.json";
    var tr = qs("trials") || (qs("set") ? "trials/" + qs("set") + ".json" : null);
    var loaded = { manifest: null, trials: null };
    function ready() { if (loaded.manifest && loaded.trials) { $("go").disabled = false; setMsg("Ready."); } }
    function tryFetch() {
      fetchJson(man).then(function (m) { loaded.manifest = m; ready(); }).catch(function () { setMsg("Choose manifest.json and the trial file below (files could not be fetched).", false); });
      if (tr) fetchJson(tr).then(function (t) { loaded.trials = t; ready(); }).catch(function () { /* file picker */ });
    }
    $("f_manifest").addEventListener("change", function (e) { readFile(e.target.files[0]).then(function (m) { loaded.manifest = m; ready(); }).catch(function (x) { setMsg(String(x), true); }); });
    $("f_trials").addEventListener("change", function (e) { readFile(e.target.files[0]).then(function (t) { loaded.trials = t; ready(); }).catch(function (x) { setMsg(String(x), true); }); });
    $("go").addEventListener("click", function () { start(loaded.manifest, loaded.trials); });
    $("begin").addEventListener("click", function () { $("intro").hidden = true; showTrial(); });
    $("submit").addEventListener("click", submit);
    $("dl").addEventListener("click", download);
    if (qs("listener")) $("listener").value = qs("listener");
    $("go").disabled = true;
    tryFetch();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})(typeof window !== "undefined" ? window : globalThis);
