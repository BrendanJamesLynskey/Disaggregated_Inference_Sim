// disagg-sim, JavaScript port of Disaggregated_Inference_Sim (SimPy).
// Same cost model, same scheduling rules, same metrics; a hand-written
// event heap stands in for SimPy's environment.
(function (root) {
    const MODELS = {
        'llama3-8b':  { name: 'Llama-3-8B',  L: 32, d: 4096, h: 32, kvh: 8, ff: 14336, V: 128256 },
        'llama3-70b': { name: 'Llama-3-70B', L: 80, d: 8192, h: 64, kvh: 8, ff: 28672, V: 128256 },
    };
    const DEVICES = {
        // idle W, dynamic pJ/FLOP and pJ/HBM-byte: illustrative, as in hardware.py
        h100:    { name: 'H100-SXM', F: 989e12,  B: 3.35e12,  M: 80e9, fe: 0.55, be: 0.8, tdp: 700, idle: 100, pjF: 1.0, pjB: 60 },
        a100:    { name: 'A100-SXM', F: 312e12,  B: 2.039e12, M: 80e9, fe: 0.55, be: 0.8, tdp: 400, idle: 60,  pjF: 1.6, pjB: 70 },
        optical: { name: 'Hypothetical optical MAC', F: 4000e12, B: 3.35e12, M: 80e9, fe: 0.4, be: 0.8, tdp: 700, idle: 180, pjF: 0.1, pjB: 60 },
    };
    const LINKS = {
        'nvlink4':  { name: 'NVLink 4',      bw: 450e9,   lat: 5e-6,  pjBit: 5 },
        'ib-ndr':   { name: 'IB NDR 400G',   bw: 50e9,    lat: 10e-6, pjBit: 15 },
        'pcie5':    { name: 'PCIe Gen5 x16', bw: 64e9,    lat: 5e-6,  pjBit: 6 },
        'eth-100g': { name: '100 GbE',       bw: 12.5e9,  lat: 20e-6, pjBit: 15 },
        'eth-25g':  { name: '25 GbE',        bw: 3.125e9, lat: 20e-6, pjBit: 15 },
    };
    const STAGES = ['prefill_queue', 'prefill', 'kv_wait', 'kv_transfer', 'decode_queue', 'decode'];
    const OWNER = { prefill_queue: 'prefill', prefill: 'prefill', kv_wait: 'kv-link',
                    kv_transfer: 'kv-link', decode_queue: 'decode', decode: 'decode' };

    function derive(m) {
        const hd = m.d / m.h, kv = m.kvh * hd;
        const ppl = 2 * m.d * m.d + 2 * m.d * kv + 3 * m.d * m.ff;
        // weightBytes: resident (both vocab tables); weightStream: read by every step (layers +
        // LM head); embRow: one embedding row, read per token looked up (corrected 2026-10-03)
        return { ...m, params: m.L * ppl + 2 * m.V * m.d, matmul: m.L * ppl + m.V * m.d,
                 weightBytes: 2 * (m.L * ppl + 2 * m.V * m.d), weightStream: 2 * (m.L * ppl + m.V * m.d),
                 embRow: 2 * m.d, kvTok: 2 * m.L * kv * 2 };
    }

    const cbrt = x => Math.sign(x) * Math.pow(Math.abs(x), 1 / 3);
    function costModel(model, dev, n, overhead, powerCap, dvfs, sMin) {
        const Fr = dev.F * dev.fe * n, Br = dev.B * dev.be * n;
        const jF = dev.pjF * 1e-12, jB = dev.pjB * 1e-12, idleW = dev.idle * n;
        const cap = Math.min(powerCap ?? Infinity, dev.tdp);       // the board limit always applies
        const budget = (cap - dev.idle) * n;
        if (budget !== null && budget <= 0) throw new Error('power cap is below idle power');
        sMin = sMin ?? 0.4;
        // DVFS three-roof step model: mirror of CostModel.step_time in hardware.py
        const t = (flops, bytes) => {
            const tc = flops / Fr, tm = bytes / Br;
            let ec = flops * jF; const em = bytes * jB;
            let s = (dvfs && tc < tm) ? Math.max(sMin, tc / tm) : 1.0;
            let bound = tc >= tm ? 'compute' : 'memory';
            if (budget !== null && (ec * s * s + em) / Math.max(tc / s, tm) > budget) {
                bound = 'power';
                const x = (budget * tm - em) / ec;
                if (x > 0 && Math.sqrt(x) * tm >= tc) s = Math.min(s, Math.sqrt(x));
                else {
                    const p = em / ec, q = -budget * tc / ec, r = Math.sqrt(q * q / 4 + p * p * p / 27);
                    s = cbrt(-q / 2 + r) + cbrt(-q / 2 - r);
                }
                s = Math.max(s, sMin);
            }
            let time = Math.max(tc / s, tm);
            ec = ec * s * s;
            if (budget !== null && (ec + em) / time > budget) time = (ec + em) / budget;
            return { flops, bytes, time: time + overhead, bound, ec, em };
        };
        return {
            idleW,
            kvCap: Math.floor((dev.M * n * 0.9 - model.weightBytes) / model.kvTok),
            prefill(lens) {
                let tok = 0, fl = 0;
                for (const s of lens) { tok += s; fl += 2 * model.L * model.d * s * (s + 1); }
                return t(2 * model.matmul * tok + fl, model.weightStream + tok * model.embRow + tok * model.kvTok);
            },
            decode(ctx) {
                let c = 0; for (const x of ctx) c += x;
                const b = ctx.length;
                // each new token attends to its context and to itself: c + b positions
                return t(2 * model.matmul * b + 4 * model.L * model.d * (c + b),
                         model.weightStream + b * model.embRow + (c + b) * model.kvTok);
            },
        };
    }

    // ── workload ───────────────────────────────────────────────────────
    function mulberry32(a) {
        return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
            t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    }
    function makeWorkload(rate, n, prompt, promptCv, output, outputCv, seed) {
        const r = mulberry32(seed || 1);
        const normal = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
        const len = (mean, cv) => {
            if (cv <= 0) return Math.round(mean);
            const s2 = Math.log(1 + cv * cv);
            const x = Math.exp(Math.log(mean) - s2 / 2 + Math.sqrt(s2) * normal());
            return Math.min(32768, Math.max(1, Math.round(x)));
        };
        const out = []; let t = 0;
        for (let i = 0; i < n; i++) {
            t += -Math.log(1 - r()) / rate;
            out.push([t, len(prompt, promptCv), len(output, outputCv)]);
        }
        return out;
    }

    // ── engine ─────────────────────────────────────────────────────────
    function simulate(cfg, rows) {
        const model = derive(MODELS[cfg.model]), dev = DEVICES[cfg.device];
        const link = { ...LINKS[cfg.link], ch: cfg.linkChannels || 1 };
        const capFor = role => (role === 'prefill' ? cfg.prefillPowerCap : role === 'decode' ? cfg.decodePowerCap : null) ?? cfg.powerCap ?? null;
        const cmFor = role => costModel(model, dev, cfg.devicesPerInstance, cfg.stepOverhead ?? 0.5e-3, capFor(role), !!cfg.dvfs);
        const cm = cmFor('colocated');
        const maxPT = cfg.maxPrefillTokens || 8192, maxB = cfg.maxDecodeBatch || 256;

        let now = 0, seq = 0, nDone = 0, nRej = 0, stop = false;
        const heap = [];
        const push = (t, f) => {
            const e = [t, seq++, f]; heap.push(e);
            let i = heap.length - 1;
            while (i > 0) { const p = (i - 1) >> 1;
                if (heap[p][0] < e[0] || (heap[p][0] === e[0] && heap[p][1] < e[1])) break;
                heap[i] = heap[p]; i = p; }
            heap[i] = e;
        };
        const pop = () => {
            const top = heap[0], last = heap.pop();
            if (heap.length) { let i = 0; const n = heap.length;
                for (;;) { let l = 2 * i + 1, r = l + 1, m = i;
                    const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
                    if (l < n && less(heap[l], m === i ? last : heap[m])) m = l;
                    if (r < n && less(heap[r], m === i ? last : heap[m])) m = r;
                    if (m === i) break; heap[i] = heap[m]; i = m; }
                heap[i] = last; }
            return top;
        };

        const reqs = rows.map(([a, p, o], i) => ({ rid: i, arrival: a, prompt: p, output: o,
            prefillStart: null, firstToken: null, kvStart: null, kvReady: null, decodeStart: null,
            finish: null, tokensOut: 0, lastToken: null, itls: [] }));
        const samples = [];

        function mkInst(role, idx) {
            return { role, name: `${role}-${idx}`, queue: [], running: [], kvUsed: 0, busy: 0, steps: 0,
                     flops: 0, bytes: 0, batchSum: 0, active: false, wake: false, cm: cmFor(role),
                     ec: 0, em: 0, peakW: 0, powerBound: 0 };
        }
        const prefill = [], decode = [], coloc = [];
        if (cfg.mode === 'disagg') {
            for (let i = 0; i < cfg.nPrefill; i++) prefill.push(mkInst('prefill', i));
            for (let i = 0; i < cfg.nDecode; i++) decode.push(mkInst('decode', i));
        } else for (let i = 0; i < cfg.nColocated; i++) coloc.push(mkInst('colocated', i));
        const insts = [...prefill, ...decode, ...coloc];
        const ls = { busy: 0, bytes: 0, transfers: 0, wait: 0, inUse: 0, queue: [], energy: 0 };

        const kvNeed = r => r.prompt + r.output;
        const load = i => i.role === 'prefill' ? i.queue.reduce((s, r) => s + r.prompt, 0) : i.queue.length + i.running.length;
        const pick = arr => arr.reduce((b, i) => load(i) < load(b) ? i : b, arr[0]);

        function finish(r) { r.finish = now; nDone++; if (nDone + nRej === reqs.length) stop = true; }
        function submit(inst, r) {
            inst.queue.push(r);
            if (!inst.active && !inst.wake) { inst.wake = true; push(now, () => { inst.wake = false; loop(inst); }); }
        }
        function step(inst, cost, batch, then) {
            inst.active = true;
            push(now + cost.time, () => {
                inst.busy += cost.time; inst.steps++; inst.flops += cost.flops; inst.bytes += cost.bytes;
                inst.batchSum += batch; inst.ec += cost.ec; inst.em += cost.em;
                const pw = inst.cm.idleW + (cost.ec + cost.em) / cost.time; if (pw > inst.peakW) inst.peakW = pw;
                if (cost.bound === 'power') inst.powerBound += cost.time;
                inst.active = false; then(); loop(inst);
            });
        }
        function firstToken(r) { r.firstToken = r.lastToken = now; r.tokensOut = 1; }
        function decodeDone(inst) {
            const still = [];
            for (const r of inst.running) {
                r.itls.push(now - r.lastToken); r.tokensOut++; r.lastToken = now;
                if (r.tokensOut >= r.output) { inst.kvUsed -= kvNeed(r); finish(r); } else still.push(r);
            }
            inst.running = still;
        }
        function decodeOnce(inst) {
            const ctx = inst.running.map(r => r.prompt + r.tokensOut);
            step(inst, inst.cm.decode(ctx), ctx.length, () => decodeDone(inst));
        }
        function loop(inst) {
            if (inst.active) return;
            if (inst.role === 'prefill') {
                if (!inst.queue.length) return;
                const batch = []; let tok = 0;
                while (inst.queue.length && (!batch.length || tok + inst.queue[0].prompt <= maxPT)) {
                    const r = inst.queue.shift(); batch.push(r); tok += r.prompt; }
                batch.forEach(r => r.prefillStart = now);
                step(inst, inst.cm.prefill(batch.map(r => r.prompt)), batch.length, () => {
                    for (const r of batch) { firstToken(r); if (r.output <= 1) finish(r); else kvRequest(r); }
                });
            } else if (inst.role === 'decode') {
                while (inst.queue.length && inst.running.length < maxB && inst.kvUsed + kvNeed(inst.queue[0]) <= cm.kvCap) {
                    const r = inst.queue.shift(); inst.kvUsed += kvNeed(r); r.decodeStart = now; inst.running.push(r); }
                if (inst.running.length) decodeOnce(inst);
            } else {
                const batch = []; let tok = 0;
                while (inst.queue.length && inst.running.length + batch.length < maxB
                       && (!batch.length || tok + inst.queue[0].prompt <= maxPT)
                       && inst.kvUsed + kvNeed(inst.queue[0]) <= cm.kvCap) {
                    const r = inst.queue.shift(); inst.kvUsed += kvNeed(r); batch.push(r); tok += r.prompt; }
                if (batch.length) {
                    batch.forEach(r => r.prefillStart = now);
                    step(inst, inst.cm.prefill(batch.map(r => r.prompt)), batch.length, () => {
                        for (const r of batch) { firstToken(r);
                            if (r.output <= 1) { inst.kvUsed -= kvNeed(r); finish(r); }
                            else { r.decodeStart = now; inst.running.push(r); } }
                    });
                } else if (inst.running.length) decodeOnce(inst);
            }
        }
        function kvRequest(r) { if (ls.inUse < link.ch) kvStart(r); else ls.queue.push(r); }
        function kvStart(r) {
            ls.inUse++; r.kvStart = now; ls.wait += now - r.firstToken;
            const nbytes = r.prompt * model.kvTok, t = link.lat + nbytes / link.bw;
            push(now + t, () => {
                ls.inUse--; ls.busy += t; ls.bytes += nbytes; ls.transfers++; r.kvReady = now;
                ls.energy += nbytes * 8 * link.pjBit * 1e-12;
                if (ls.queue.length) kvStart(ls.queue.shift());
                submit(pick(decode), r);
            });
        }
        // arrivals
        for (const r of reqs) push(r.arrival, () => {
            const pool = cfg.mode === 'disagg' ? decode : coloc;
            if (kvNeed(r) > cm.kvCap) { nRej++; r.rejected = true; if (nDone + nRej === reqs.length) stop = true; return; }
            submit(pick(cfg.mode === 'disagg' ? prefill : pool), r);
        });
        // sampler (passive probe)
        const dt = cfg.sampleDt || 0.25;
        const sample = () => {
            const row = { t: now, linkQ: ls.queue.length, prefillQ: 0, decodeQ: 0, running: 0 };
            for (const i of insts) { if (i.role === 'prefill') row.prefillQ += i.queue.length;
                else { row.decodeQ += i.queue.length; row.running += i.running.length; } }
            samples.push(row); push(now + dt, sample);
        };
        push(0, sample);

        while (heap.length && !stop) { const e = pop(); now = e[0]; e[2](); }
        return { cfg, model, cm, reqs, insts, link: ls, linkCh: link.ch, horizon: now, samples };
    }

    // ── metrics (mirror of metrics.py) ─────────────────────────────────
    function pctSorted(s, p) {
        if (!s.length) return NaN;
        const k = (s.length - 1) * p / 100, lo = Math.floor(k), hi = Math.ceil(k);
        return s[lo] + (s[hi] - s[lo]) * (k - lo);
    }
    const percentile = (xs, p) => pctSorted(Float64Array.from(xs).sort(), p);
    const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
    const dist = xs => { const s = Float64Array.from(xs).sort();     // sort once for all three
                         return { mean: mean(xs), p50: pctSorted(s, 50), p90: pctSorted(s, 90), p99: pctSorted(s, 99) }; };
    function stages(r) {
        const hand = r.kvReady !== null ? r.kvReady : r.firstToken;
        const s = { prefill_queue: r.prefillStart - r.arrival, prefill: r.firstToken - r.prefillStart,
                    kv_wait: 0, kv_transfer: 0, decode_queue: 0, decode: 0 };
        if (r.kvStart !== null) { s.kv_wait = r.kvStart - r.firstToken; s.kv_transfer = r.kvReady - r.kvStart; }
        if (r.decodeStart !== null) { s.decode_queue = r.decodeStart - hand; s.decode = r.finish - r.decodeStart; }
        return s;
    }
    function summarise(res) {
        const cfg = res.cfg, H = res.horizon;
        const done = res.reqs.filter(r => r.finish !== null).sort((a, b) => a.arrival - b.arrival);
        const steady = done.slice(Math.floor(done.length * (cfg.warmupFrac ?? 0.1)));
        const ttft = steady.map(r => r.firstToken - r.arrival);
        const tpot = steady.filter(r => r.output >= 2).map(r => (r.finish - r.firstToken) / (r.output - 1));
        const itl = []; steady.forEach(r => { for (const x of r.itls) itl.push(x); });
        const e2e = steady.map(r => r.finish - r.arrival);
        const met = steady.filter(r => (r.firstToken - r.arrival) <= cfg.ttftSlo &&
            (r.output < 2 || (r.finish - r.firstToken) / (r.output - 1) <= cfg.tpotSlo));
        const win = steady.length > 1 ? steady[steady.length - 1].arrival - steady[0].arrival : NaN;
        const util = {}; res.insts.forEach(i => util[i.name] = i.busy / H);
        util['kv-link'] = res.link.busy / (res.linkCh * H);
        const st = {}; STAGES.forEach(k => st[k] = mean(steady.map(r => stages(r)[k])));
        const e2eMean = mean(e2e);
        const waits = STAGES.filter(k => k !== 'decode');
        const hot = waits.reduce((a, b) => st[b] > st[a] ? b : a, waits[0]);
        let owner = OWNER[hot]; if (cfg.mode === 'colocated' && owner !== 'kv-link') owner = 'colocated';
        const pool = Object.keys(util).filter(k => k.startsWith(owner));
        const hotRes = pool.reduce((a, b) => util[b] > util[a] ? b : a, pool[0]);
        const eff = {}; res.insts.forEach(i => eff[i.name] = {
            mfu: i.flops / (H * res.cfgDev().F * res.cfg.devicesPerInstance),
            mbu: i.bytes / (H * res.cfgDev().B * res.cfg.devicesPerInstance),
            batch: i.steps ? i.batchSum / i.steps : 0 });
        // energy: static power for the whole run + dynamic work + link (mirror of energy_report)
        let st_ = 0, ce = 0, me = 0; const perE = {};
        res.insts.forEach(i => { const s_ = i.cm.idleW * H; st_ += s_; ce += i.ec; me += i.em;
            perE[i.name] = { avgW: (s_ + i.ec + i.em) / H, peakW: i.peakW, powerBoundFrac: i.busy ? i.powerBound / i.busy : 0 }; });
        const outTok = done.reduce((s, r) => s + r.output, 0), totJ = st_ + ce + me + res.link.energy;
        const energy = { totalJ: totJ, avgW: totJ / H, jPerTok: totJ / outTok, tokPerJ: outTok / totJ,
            breakdown: { static: st_ / totJ, compute: ce / totJ, memory: me / totJ, link: res.link.energy / totJ }, perInstance: perE };
        return {
            mode: cfg.mode, completed: done.length, measured: steady.length, energy,
            rejected: res.reqs.length - done.length,
            ttft: dist(ttft), tpot: dist(tpot), itl: dist(itl), e2e: dist(e2e),
            tokPerS: done.reduce((s, r) => s + r.output, 0) / H, reqPerS: done.length / H,
            goodput: win > 0 ? met.length / win : NaN, sloAttain: steady.length ? met.length / steady.length : NaN,
            util, eff, stages: st, stageShare: Object.fromEntries(STAGES.map(k => [k, st[k] / e2eMean])),
            hot: { stage: hot, resource: hotRes, util: util[hotRes] },
            kvLink: { GB: res.link.bytes / 1e9, meanWaitMs: 1e3 * res.link.wait / Math.max(1, res.link.transfers) },
            raw: { ttft, itl }, samples: res.samples, horizon: H,
        };
    }
    function run(cfg, rows) {
        const res = simulate(cfg, rows);
        res.cfgDev = () => DEVICES[cfg.device];
        return summarise(res);
    }
    root.DisaggSim = { MODELS, DEVICES, LINKS, STAGES, derive, costModel, makeWorkload, simulate, summarise, run, percentile };
})(typeof window !== 'undefined' ? window : globalThis);
