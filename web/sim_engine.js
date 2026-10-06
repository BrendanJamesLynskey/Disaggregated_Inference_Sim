// disagg-sim, JavaScript port of Disaggregated_Inference_Sim (SimPy).
// Same cost model, same scheduling rules, same metrics; a hand-written
// event heap stands in for SimPy's environment.
// 2026-10-04: heterogeneous pools, FFT-mixing models (Hyena, hybrid, block-circulant,
// distilled decode), the optical transform engine, KV hand-off compression (in transit or
// at the GPU) and the PPA functions, all bit-exact with the Python package (tested).
// 2026-10-05: the Causal Encoder-Decoder (CED) option of DeepSeek-V4.1-Flash (arXiv:2609.19969):
// cedE encoder layers, a cedW-token replay through the decoder on prefill or (SGLang RFC #39963)
// as the decode instance's first step. Off unless a model or cfg turns it on (tested bit-exact).
// 2026-10-06 (brief 20A1): colocated batching policies (prefill-priority, decode-priority, Sarathi-Serve
// chunked prefill with a token budget), KV policies (oracle reservation, pow2, max, paged blocks with
// recompute or swap preemption), prefix caching (LRU cache of shared prompt segments) and closed-loop
// sessions; sim.ScheduledInstance, bit-exact (tested). Off unless cfg turns a lever on.
(function (root) {
    // mixer: 'attention' | 'hyena' | 'hybrid'; see hardware.ModelSpec for the fields
    const MODELS = {
        'llama3-8b':  { name: 'Llama-3-8B',  L: 32, d: 4096, h: 32, kvh: 8, ff: 14336, V: 128256 },
        'llama3-70b': { name: 'Llama-3-70B', L: 80, d: 8192, h: 64, kvh: 8, ff: 28672, V: 128256 },
        'llama3-8b-hyena': { name: 'Llama-3-8B-shape Hyena-2', L: 32, d: 4096, h: 32, kvh: 8, ff: 14336, V: 128256,
                             mixer: 'hyena' },
        'llama3-8b-hyena-dist': { name: 'Llama-3-8B-shape Hyena-2 (distilled decode)', L: 32, d: 4096, h: 32, kvh: 8,
                                  ff: 14336, V: 128256, mixer: 'hyena', decodeStyle: 'distilled' },
        'llama3-8b-hybrid': { name: 'Llama-3-8B-shape hybrid 1:3', L: 32, d: 4096, h: 32, kvh: 8, ff: 14336, V: 128256,
                              mixer: 'hybrid' },
        'llama3-8b-hyena-circ': { name: 'Llama-3-8B-shape Hyena-2 + block-circulant 256', L: 32, d: 4096, h: 32, kvh: 8,
                                  ff: 14336, V: 128256, mixer: 'hyena', circ: 256 },
        // CED proxies: half encoder, half decoder (DeepSeek-V4.1-Flash splits 40 layers 20/20); illustrative
        'llama3-8b-ced':  { name: 'Llama-3-8B-shape CED 16+16',  L: 32, d: 4096, h: 32, kvh: 8, ff: 14336, V: 128256, cedE: 16 },
        'llama3-70b-ced': { name: 'Llama-3-70B-shape CED 40+40', L: 80, d: 8192, h: 64, kvh: 8, ff: 28672, V: 128256, cedE: 40 },
        // brief 20A1 validation shapes (Hugging Face configs; OPT's MLP as d_ff = 13653, see hardware.py)
        'mistral-7b': { name: 'Mistral-7B', L: 32, d: 4096, h: 32, kvh: 8, ff: 14336, V: 32000 },
        'yi-34b':     { name: 'Yi-34B',     L: 60, d: 7168, h: 56, kvh: 8, ff: 20480, V: 64000 },
        'opt-13b':    { name: 'OPT-13B',    L: 40, d: 5120, h: 40, kvh: 40, ff: 13653, V: 50272 },
    };
    const ENOB_REQUIRED = { bf16: 11, int8: 8, fp8: 7 };
    // Fourier-optical transform engine (illustrative; hardware.TransformEngine)
    const ENGINE = { sps: 1e12, enob: 8, fomDac: 10, fomAdc: 20, laser: 10, tuning: 10, maskValues: 2000000,
                     maskRate: 1031, detection: 'coherent', overlap: false };
    const DEVICES = {
        // idle W, dynamic pJ/FLOP and pJ/HBM-byte: illustrative, as in hardware.py. Bandwidths are
        // written as hardware.py computes them (2.039 * TB is not the literal 2.039e12).
        h100:    { name: 'H100-SXM', F: 989e12,  B: 3.35e12,  M: 80e9, fe: 0.55, be: 0.8, tdp: 700, idle: 100, pjF: 1.0, pjB: 60 },
        a100:    { name: 'A100-SXM', F: 312e12,  B: 2.039 * 1e12, M: 80e9, fe: 0.55, be: 0.8, tdp: 400, idle: 60,  pjF: 1.6, pjB: 70 },
        'a100-40g': { name: 'A100-SXM-40GB', F: 312e12, B: 1.555 * 1e12, M: 40e9, fe: 0.55, be: 0.8, tdp: 400, idle: 60, pjF: 1.6, pjB: 70 },
        optical: { name: 'Hypothetical optical MAC', F: 4000e12, B: 3.35e12, M: 80e9, fe: 0.4, be: 0.8, tdp: 700, idle: 180, pjF: 0.1, pjB: 60 },
        // optical transform engine co-packaged with an H100-class / A100-class digital part
        'optical-fft':       { name: 'Optical-FFT + H100-class', F: 989e12, B: 3.35e12, M: 80e9, fe: 0.55, be: 0.8, tdp: 700, idle: 100, pjF: 1.0, pjB: 60, transform: ENGINE },
        'optical-fft-small': { name: 'Optical-FFT + A100-class', F: 312e12, B: 2.039 * 1e12, M: 80e9, fe: 0.55, be: 0.8, tdp: 400, idle: 60, pjF: 1.6, pjB: 70, transform: ENGINE },
    };
    const LINKS = {
        'nvlink4':  { name: 'NVLink 4',      bw: 450e9,   lat: 5e-6,  pjBit: 5 },
        'ib-ndr':   { name: 'IB NDR 400G',   bw: 50e9,    lat: 10e-6, pjBit: 15 },
        'pcie5':    { name: 'PCIe Gen5 x16', bw: 64e9,    lat: 5e-6,  pjBit: 6 },
        'pcie4':    { name: 'PCIe Gen4 x16', bw: 32e9,    lat: 5e-6,  pjBit: 6 },
        'eth-100g': { name: '100 GbE',       bw: 12.5e9,  lat: 20e-6, pjBit: 15 },
        'eth-25g':  { name: '25 GbE',        bw: 3.125e9, lat: 20e-6, pjBit: 15 },
        // photonic interconnect, not Fourier optics: illustrative round numbers
        'cpo-optical': { name: 'Co-packaged optics (illustrative)', bw: 200e9, lat: 5e-6, pjBit: 3 },
    };
    // KV hand-off compression (hardware.KV_PRESETS): ratio = bytes in / out, ops per BF16 value
    const KV_PRESETS = {
        'none':        { name: 'none', ratio: 1.0, ops: 0.0, fft: false },
        'fp8':         { name: 'fp8', ratio: 2.0, ops: 1.0, fft: false },
        'fp4-block':   { name: 'fp4-block', ratio: 64 / 17, ops: 2.0, fft: false },
        'freq-keep-k': { name: 'freq-keep-k', ratio: 2.0, ops: 1.0, fft: true },
    };
    const TRANSIT = { opsPerByte: 1.6, pjBit: 1.0, lat: 1e-6, nativeFft: true };
    const STAGES = ['prefill_queue', 'prefill', 'kv_wait', 'kv_transfer', 'decode_queue', 'decode'];
    const OWNER = { prefill_queue: 'prefill', prefill: 'prefill', kv_wait: 'kv-link',
                    kv_transfer: 'kv-link', decode_queue: 'decode', decode: 'decode' };

    // integer log2 for powers of two, as hardware.rfft_flops uses
    const ilog2 = n => 31 - Math.clz32(n);
    const rfft = n => 2.5 * n * ilog2(n);
    const pow2AtLeast = n => 2 ** (32 - Math.clz32(n - 1));          // 1 << (n - 1).bit_length()
    const idiv = (a, b) => Math.floor(a / b);
    // FLOPs by class (hardware.Ops); add() keeps Python's field-by-field order
    const ops = (o = {}) => ({ dense: 0, attention: 0, transform: 0, spectral: 0, other: 0, ...o });
    const add = (a, b) => { a.dense += b.dense; a.attention += b.attention; a.transform += b.transform;
                            a.spectral += b.spectral; a.other += b.other; return a; };
    const total = o => o.dense + o.attention + o.transform + o.spectral + o.other;
    const optical = o => o.transform + o.spectral;
    const digital = o => o.dense + o.attention + o.other;

    function derive(m0) {
        const m = { mixer: 'attention', attnEvery: 4, order: 2, short: 3, circ: 0, decodeStyle: 'direct', ds: 16,
                    actFormat: 'bf16', lmHead: 'all', cedE: 0, cedW: 128, cedOn: 'prefill', ...m0 };
        if (m.cedE) {
            if (m.mixer !== 'attention') throw new Error("ced_encoder_layers is modelled for mixer='attention' only");
            if (!(m.cedE > 0 && m.cedE < m.L)) throw new Error('ced_encoder_layers must leave at least one decoder layer');
            if (m.cedW < 1) throw new Error('ced_replay must be at least 1 token');
            if (m.cedOn !== 'prefill' && m.cedOn !== 'decode') throw new Error(`unknown ced_replay_on ${m.cedOn}`);
        }
        const hd = m.d / m.h, kv = m.kvh * hd;
        const ppl = 2 * m.d * m.d + 2 * m.d * kv + 3 * m.d * m.ff;
        const isAttn = i => m.mixer === 'attention' || (m.mixer === 'hybrid' && i % m.attnEvery === 0);
        let nAttn = 0; for (let i = 0; i < m.L; i++) if (isAttn(i)) nAttn++;
        const nHy = m.L - nAttn, N = m.order, d = m.d, k = m.circ;
        const mats = [[(N + 1) * d, d], [d, d], [m.ff, d], [m.ff, d], [d, m.ff]];
        let hyp = 0; for (const [r, c] of mats) hyp += k ? idiv(r * c, k) : r * c;
        const layers = nAttn * ppl + nHy * hyp;
        const conv = m.decodeStyle === 'direct' ? nHy * N * d : 0;
        const kvTok = (2 * nAttn * kv + conv) * 2;
        const state = m.decodeStyle === 'distilled' ? nHy * N * d * m.ds * 4.0 : 0;
        let pairs = 0;
        if (m.mixer !== 'attention') { let per = N * d; if (k) for (const [r, c] of mats) per += Math.max(r, c); pairs = nHy * per; }
        // weightBytes: resident (both vocab tables); weightStream: read by every step (layers +
        // LM head); embRow: one embedding row, read per token looked up (corrected 2026-10-03)
        const mm = { ...m, isAttn, nAttn, nHy, mats, layers, ppl, params: layers + 2 * m.V * m.d, matmul: layers + m.V * m.d,
                     weightBytes: 2 * (layers + 2 * m.V * m.d), weightStream: 2 * (layers + m.V * m.d),
                     embRow: 2 * m.d, kvTok, state, cacheUnit: kvTok ? kvTok : state, pairs,
                     transformer: m.mixer === 'attention' };
        // CED: parameters a prompt token touches (encoder + the decoder's K/V projections), and what a
        // prefill-only instance holds when the replay runs on decode (+ the input embedding)
        mm.cedPrompt = m.cedE * ppl + (m.L - m.cedE) * 2 * m.d * kv;
        mm.cedResident = mm.cedPrompt + m.V * m.d;
        mm.cedAttn = (layers_, s, w) => 2 * layers_ * m.d * w * (2 * s - w + 1);
        mm.units = (p, o) => kvTok ? p + o : 1;
        mm.handoff = p => state ? state : p * kvTok;
        const dense = (r, c, tokens) => {
            if (!k) return ops({ dense: 2.0 * r * c * tokens });
            const bins = idiv(k, 2) + 1;
            return ops({ transform: tokens * (idiv(c, k) * rfft(k) + idiv(r, k) * rfft(k)),
                         spectral: tokens * idiv(r, k) * idiv(c, k) * bins * 6.0,
                         other: tokens * idiv(r, k) * (idiv(c, k) - 1) * bins * 2.0 });
        };
        const mlp = tokens => { const o = ops(); for (const [r, c] of [[m.ff, d], [m.ff, d], [d, m.ff]]) add(o, dense(r, c, tokens)); return o; };
        const distinct = lens => [...new Set(lens.map(s => pow2AtLeast(2 * s)))].sort((a, b) => a - b);
        mm.prefillOps = lens => {
            let tokens = 0; for (const s of lens) tokens += s;
            const o = ops();
            for (let i = 0; i < m.L; i++) {
                if (isAttn(i)) {
                    const a = ops({ dense: 2.0 * tokens * (2 * d * d + 2 * d * kv) });
                    for (const s of lens) a.attention += 2.0 * d * s * (s + 1);
                    add(o, a);
                } else {
                    const h = ops();
                    add(h, dense((N + 1) * d, d, tokens)); add(h, dense(d, d, tokens));
                    h.other += tokens * 2.0 * m.short * (N + 1) * d;
                    h.other += tokens * N * d;
                    for (const s of lens) { const f = pow2AtLeast(2 * s);
                        h.transform += N * d * 2 * rfft(f); h.spectral += N * d * (idiv(f, 2) + 1) * 6.0; }
                    add(o, h);
                }
                add(o, mlp(tokens));
            }
            add(o, ops({ dense: 2.0 * m.V * d * (m.lmHead === 'all' ? tokens : lens.length) }));
            return o;
        };
        // the weight-matrix parts depend only on the batch: computed once per batch size (same
        // values, added in the same order: ModelSpec._decode_parts)
        const partsCache = new Map();
        const parts = b => { let p = partsCache.get(b);
            if (!p) { p = [dense((N + 1) * d, d, b), dense(d, d, b), mlp(b), ops({ dense: 2.0 * m.V * d * b })]; partsCache.set(b, p); }
            return p; };
        mm.decodeOps = (ctx, b) => {
            const [pin, pout, pm, head] = parts(b);
            const o = ops();
            for (let i = 0; i < m.L; i++) {
                if (isAttn(i)) add(o, ops({ dense: 2.0 * b * (2 * d * d + 2 * d * kv), attention: 4.0 * d * (ctx + b) }));
                else {
                    add(o, pin); add(o, pout);
                    if (m.decodeStyle === 'direct') o.other += N * d * 2.0 * (ctx + b);
                    else o.other += N * d * 8.0 * m.ds * b;
                }
                add(o, pm);
            }
            add(o, head);
            return o;
        };
        mm.maskValues = lens => {
            if (mm.transformer) return 0;
            let v = 0; for (const f of distinct(lens)) v += nHy * N * d * (idiv(f, 2) + 1);
            if (k) { let c = 0; for (const [r, cc] of mats) c += idiv(r, k) * idiv(cc, k) * (idiv(k, 2) + 1); v += nHy * c; }
            return v;
        };
        mm.spectrumBytes = lens => {
            if (mm.transformer) return 0.0;
            let v = 0; for (const f of distinct(lens)) v += nHy * N * d * (idiv(f, 2) + 1);
            return v * 4.0;
        };
        return mm;
    }
    // device with its transform engine's overrides and FFT efficiency applied
    function deviceFor(key, cfg) {
        const base = DEVICES[key];
        if (!base) throw new Error(`unknown device ${key}`);
        const dev = { fftEff: 1.0, ...base };
        if (cfg && cfg.fftEff != null) dev.fftEff = cfg.fftEff;
        if (dev.transform && cfg && cfg.engine) {
            const e = { ...dev.transform, ...cfg.engine };
            if (cfg.engine.staticW != null) { e.laser = cfg.engine.staticW / 2; e.tuning = cfg.engine.staticW / 2; }
            dev.transform = e;
        }
        return dev;
    }
    const passes = (eng, fmt) => { const k = 4 ** Math.max(0, ENOB_REQUIRED[fmt] - eng.enob);
                                   return eng.detection === 'intensity' ? 2 * k : k; };
    const pjPair = eng => eng.fomDac * 2 ** eng.enob * 1e-3 + eng.fomAdc * 2 ** eng.enob * 1e-3;

    const cbrt = x => Math.sign(x) * Math.pow(Math.abs(x), 1 / 3);
    function costModel(model, dev, n, overhead, powerCap, dvfs, sMin, prefillOnly) {
        // CED with the replay on decode: a prefill-pool instance holds and runs the encoder only
        const encoderOnly = !!prefillOnly && model.cedE > 0 && model.cedOn === 'decode';
        const Fr = dev.F * dev.fe * n, Br = dev.B * dev.be * n;
        const jF = dev.pjF * 1e-12, jB = dev.pjB * 1e-12, idleW = dev.idle * n;
        const eng = dev.transform || null;
        const optStaticW = eng ? (eng.laser + eng.tuning) * n : 0.0;
        const cap = Math.min(powerCap ?? Infinity, dev.tdp);       // the board limit always applies
        const budget = (cap - dev.idle) * n;
        if (budget !== null && budget <= 0) throw new Error('power cap is below idle power');
        sMin = sMin ?? 0.4;
        const fftEff = dev.fftEff ?? 1.0;
        const digitalFlops = o => fftEff === 1.0 ? total(o) : digital(o) + optical(o) / fftEff;
        // DVFS three-roof step model: mirror of CostModel.step_time_raw in hardware.py
        const raw = (flops, bytes) => {
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
            return { time, bound, ec, em };
        };
        const t = (flops, bytes) => {
            const r = raw(flops, bytes);
            return { flops, bytes, time: r.time + overhead, bound: r.bound, ec: r.ec, em: r.em, oj: 0, oflops: 0 };
        };
        const free = dev.M * n * 0.9 - (encoderOnly ? model.cedResident * 2 : model.weightBytes);
        const kvCap = free > 0 ? Math.floor(free / model.cacheUnit) : null;
        // the transform engine's integer counts: (passes, conversions, rewrites, optical seconds)
        const opticalTerms = (lens, tokens) => {
            const k = passes(eng, model.actFormat), conversions = model.pairs * tokens * k;
            const vals = model.maskValues(lens), q = Math.floor(vals / eng.maskValues);
            const rewrites = q * eng.maskValues < vals ? q + 1 : q;
            return [k, conversions, rewrites, conversions / eng.sps + rewrites / eng.maskRate];
        };
        // CED prefill: CostModel._prefill_ced (same integer FLOPs, same float order for bytes)
        const cedPrefill = lens => {
            const enc = model.cedE, dec = model.L - model.cedE;
            let tok = 0; for (const s of lens) tok += s;
            let flops = 2 * model.cedPrompt * tok;
            let att = 0; for (const s of lens) att += 2 * enc * model.d * s * (s + 1);
            flops += att;
            if (encoderOnly) {
                const nbytes = model.cedPrompt * 2 + tok * model.embRow;
                return t(flops, nbytes + tok * model.kvTok);
            }
            let rt = 0, ra = 0;
            for (const s of lens) { const w = Math.min(s, model.cedW); rt += w; ra += model.cedAttn(dec, s, w); }
            flops += 2 * dec * model.ppl * rt;
            flops += ra;
            flops += 2 * model.V * model.d * (model.lmHead === 'all' ? rt : lens.length);
            return t(flops, model.weightStream + tok * model.embRow + tok * model.kvTok);
        };
        return {
            idleW, optStaticW, kvCap, opticalTerms, encoderOnly,
            fits: free > 0,
            // CED, replay on decode: running rows decode while new rows replay w prompt tokens (CostModel.ced_step)
            cedStep(ctx, b, replay) {
                let rt = 0, ra = 0, rs = 0;
                for (const [s, w] of replay) { rt += w; ra += model.cedAttn(model.L, s, w); rs += s; }
                let flops = 2 * model.matmul * b + 4 * model.L * model.d * (ctx + b);
                flops += 2 * model.layers * rt;
                flops += 2 * model.V * model.d * (model.lmHead === 'all' ? rt : replay.length);
                flops += ra;
                let nbytes = model.weightStream + (b + rt) * model.embRow + (ctx + b) * model.kvTok;
                nbytes += rs * model.kvTok;
                return t(flops, nbytes);
            },
            prefill(lens) {
                let tok = 0, fl = 0;
                if (model.cedE) return cedPrefill(lens);
                if (!model.transformer) {
                    for (const s of lens) tok += s;
                    const o = model.prefillOps(lens);
                    let nbytes = model.weightStream + tok * model.embRow + tok * model.kvTok;
                    if (model.state) nbytes += lens.length * model.state;
                    if (!eng) return t(digitalFlops(o), nbytes + model.spectrumBytes(lens));
                    const [, conversions, , tOpt] = opticalTerms(lens, tok);
                    const flops = digital(o);
                    const r = raw(flops, nbytes);
                    const time = eng.overlap ? Math.max(r.time, tOpt) : r.time + tOpt;
                    const oj = conversions * pjPair(eng) * 1e-12;
                    return { flops, bytes: nbytes, time: time + overhead, bound: tOpt > r.time ? 'optical' : r.bound,
                             ec: r.ec, em: r.em, oj, oflops: optical(o) };
                }
                for (const s of lens) { tok += s; fl += 2 * model.L * model.d * s * (s + 1); }
                const mmf = model.lmHead === 'all' ? 2 * model.matmul * tok : 2 * model.layers * tok + 2 * model.V * model.d * lens.length;
                return t(mmf + fl, model.weightStream + tok * model.embRow + tok * model.kvTok);
            },
            // decode rows plus prefill chunks [p0, c, last] in one pass (CostModel.step_mixed)
            mixed(ctx, b, chunks) {
                let flops = 2 * model.matmul * b + 4 * model.L * model.d * (ctx + b);
                let tokens = b, kv = ctx + b;
                for (const [p0, c, last] of chunks) {
                    if (model.lmHead === 'all') flops += 2 * model.matmul * c;
                    else flops += 2 * model.layers * c + (last ? 2 * model.V * model.d : 0);
                    flops += 2 * model.L * model.d * c * (2 * p0 + c + 1);
                    tokens += c; kv += p0 + c;
                }
                return t(flops, model.weightStream + tokens * model.embRow + kv * model.kvTok);
            },
            decodeSum(c, b) {
                return t(2 * model.matmul * b + 4 * model.L * model.d * (c + b),
                         model.weightStream + b * model.embRow + (c + b) * model.kvTok);
            },
            decode(ctx) {
                let c = 0; for (const x of ctx) c += x;
                const b = ctx.length;
                if (!model.transformer) {
                    let nbytes = model.weightStream + b * model.embRow + (c + b) * model.kvTok;
                    if (model.state) nbytes += b * model.state;
                    return t(digitalFlops(model.decodeOps(c, b)), nbytes);
                }
                // each new token attends to its context and to itself: c + b positions
                return t(2 * model.matmul * b + 4 * model.L * model.d * (c + b),
                         model.weightStream + b * model.embRow + (c + b) * model.kvTok);
            },
            // endpoint KV compression: an elementwise pass after the prefill step
            compress(cost, opsN, nbytes) {
                const tc = opsN / Fr, tm = nbytes / Br, ec = opsN * jF, em = nbytes * jB;
                return { ...cost, flops: cost.flops + opsN, bytes: cost.bytes + nbytes, time: cost.time + Math.max(tc, tm),
                         ec: cost.ec + ec, em: cost.em + em };
            },
        };
    }
    // operations to compress nbytes of hand-off (hardware.KVTransit.ops)
    function transitOps(tr, nbytes, promptLen, kvBytes, onGpu) {
        const values = nbytes / kvBytes, c = tr.preset;
        let per = c.ops;
        if (c.fft && (onGpu || !tr.nativeFft)) { const n = pow2AtLeast(promptLen); per = per + 2 * rfft(n) / n; }
        return per * values;
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
        const model = derive({ ...MODELS[cfg.model], ...(cfg.lmHead ? { lmHead: cfg.lmHead } : {}),
                               ...(cfg.cedEncoderLayers != null ? { cedE: cfg.cedEncoderLayers } : {}),
                               ...(cfg.cedReplay != null ? { cedW: cfg.cedReplay } : {}),
                               ...(cfg.cedReplayOn ? { cedOn: cfg.cedReplayOn } : {}) });
        const cedOnDecode = model.cedE > 0 && model.cedOn === 'decode';
        if (cedOnDecode && cfg.mode !== 'disagg') throw new Error("ced_replay_on='decode' splits prefill across the two pools: mode='disagg' only");
        const link = { ...LINKS[cfg.link], ch: cfg.linkChannels || 1 };
        const capFor = role => (role === 'prefill' ? cfg.prefillPowerCap : role === 'decode' ? cfg.decodePowerCap : null) ?? cfg.powerCap ?? null;
        // heterogeneous pools: prefillDevice / decodeDevice (keys) default to cfg.device
        const poolOf = role => ({
            dev: deviceFor((role === 'prefill' ? cfg.prefillDevice : role === 'decode' ? cfg.decodeDevice : null) ?? cfg.device, cfg),
            n: (role === 'prefill' ? cfg.prefillDevicesPerInstance : role === 'decode' ? cfg.decodeDevicesPerInstance : null) ?? cfg.devicesPerInstance });
        const cmFor = role => { const p = poolOf(role);
            const c = costModel(model, p.dev, p.n, cfg.stepOverhead ?? 0.5e-3, capFor(role), !!cfg.dvfs, undefined, role === 'prefill');
            if (!c.fits) throw new Error(`${role} pool: ${model.name} does not fit on ${p.n}x ${p.dev.name}`);
            return { ...c, dev: p.dev, n: p.n }; };
        // KV hand-off compression: { preset, where, opsPerByte, pjBit, lat, nativeFft }
        let tr = null;
        if (cfg.kvCompress) {
            if (cfg.mode !== 'disagg') throw new Error("kv compression needs mode 'disagg'");
            const preset = { ...KV_PRESETS[cfg.kvCompress] };
            if (cfg.kvKeep != null) preset.ratio = 1 / cfg.kvKeep;
            tr = { ...TRANSIT, preset, where: cfg.kvCompressAt || 'transit',
                   ...(cfg.transitOpsPerByte != null ? { opsPerByte: cfg.transitOpsPerByte } : {}),
                   ...(cfg.transitPjPerBit != null ? { pjBit: cfg.transitPjPerBit } : {}),
                   ...(cfg.transitNativeFft != null ? { nativeFft: cfg.transitNativeFft } : {}) };
        }
        const maxPT = cfg.maxPrefillTokens || 8192, maxB = cfg.maxDecodeBatch || 256;
        // brief 20A1 levers (sim.ScheduledInstance): any of them on switches the colocated instances over
        const policy = cfg.batchPolicy || 'prefill-priority', kvPolicy = cfg.kvPolicy || 'oracle';
        const sched = policy !== 'prefill-priority' || cfg.maxNumBatchedTokens != null || kvPolicy !== 'oracle' || !!cfg.prefixCaching;
        const preemptMode = cfg.preemption || 'recompute';
        if (sched) {
            if (!['prefill-priority', 'decode-priority', 'chunked'].includes(policy)) throw new Error(`unknown batch_policy ${policy}`);
            if (!['oracle', 'pow2', 'max', 'paged'].includes(kvPolicy)) throw new Error(`unknown kv_policy ${kvPolicy}`);
            if (preemptMode !== 'recompute' && preemptMode !== 'swap') throw new Error(`unknown preemption ${preemptMode}`);
            if (cfg.mode !== 'colocated') throw new Error("batch_policy, max_num_batched_tokens, kv_policy and prefix_caching are modelled for mode='colocated' only");
            if (!model.transformer || model.cedE) throw new Error('the brief-20A1 levers are modelled for attention models without CED only');
            if (cfg.prefixCaching && kvPolicy === 'paged' && preemptMode === 'swap') throw new Error("prefix caching with swap preemption is not modelled: use preemption='recompute'");
        }
        const hostLink = LINKS[cfg.hostLink || 'pcie5'];
        const maxSeqLen = cfg.maxSeqLen ?? 2048;

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

        const reqs = rows.map(([a, p, o, x], i) => ({ rid: i, arrival: a, prompt: p, output: o,
            prefillStart: null, firstToken: null, prefillDone: null, kvStart: null, kvReady: null, decodeStart: null,
            finish: null, tokensOut: 0, lastToken: null, itls: [],
            // shared prefixes and closed-loop sessions (Request.prefix / emit_key / after / think)
            prefix: x ? x.chain : [], emit: x ? x.emit : null, after: x && x.after != null ? x.after : null, think: x ? x.think : 0,
            // ScheduledInstance state
            target: 0, done: 0, baseOut: 0, alloc: 0, nodes: [], covered: 0, swappedUnits: 0 }));
        const successor = new Map();
        for (const r of reqs) if (r.after !== null) successor.set(r.after, r);
        const samples = [];

        function mkInst(role, idx) {
            const inst = { role, name: `${role}-${idx}`, queue: [], running: [], kvUsed: 0, busy: 0, steps: 0,
                     flops: 0, bytes: 0, batchSum: 0, active: false, wake: false, cm: cmFor(role),
                     ec: 0, em: 0, peakW: 0, powerBound: 0, oj: 0, oflops: 0, opticalBound: 0 };
            if (sched && role === 'colocated') {
                const paged = kvPolicy === 'paged', blk = paged ? (cfg.kvBlockSize ?? 16) : 1;
                if (blk < 1 || (cfg.maxNumBatchedTokens != null && cfg.maxNumBatchedTokens < 1)) throw new Error('kv_block_size and max_num_batched_tokens must be at least 1');
                const cap = Math.floor(inst.cm.kvCap / blk);
                Object.assign(inst, { budget: cfg.maxNumBatchedTokens != null ? cfg.maxNumBatchedTokens : maxPT, paged, blk, cap,
                    watermark: paged ? Math.trunc((cfg.kvWatermark ?? 0.01) * cap) : 0, swap: paged && preemptMode === 'swap',
                    cache: cfg.prefixCaching ? { nodes: new Map(), seq: 0, evictedTokens: 0, evictedUnits: 0 } : null,
                    swapped: [], swapT: 0.0, preemptions: 0, recomputeTokens: 0, swapOut: 0.0, swapIn: 0.0, swapTime: 0.0,
                    swapJ: 0.0, promptTokens: 0, hitTokens: 0, computedTokens: 0, runArea: 0.0, tokArea: 0.0, allocArea: 0.0 });
            }
            return inst;
        }
        const prefill = [], decode = [], coloc = [];
        if (cfg.mode === 'disagg') {
            for (let i = 0; i < cfg.nPrefill; i++) prefill.push(mkInst('prefill', i));
            for (let i = 0; i < cfg.nDecode; i++) decode.push(mkInst('decode', i));
        } else for (let i = 0; i < cfg.nColocated; i++) coloc.push(mkInst('colocated', i));
        const insts = [...prefill, ...decode, ...coloc];
        const front = cfg.mode === 'disagg' ? decode[0] : coloc[0];      // admission: the decode pool's capacity
        const ls = { busy: 0, bytes: 0, transfers: 0, wait: 0, inUse: 0, queue: [], energy: 0,
                     handoff: 0, transitJ: 0, transitBound: 0 };

        const kvNeed = r => model.units(r.prompt, r.output);
        const load = i => {
            if (i.role === 'prefill') return i.queue.reduce((s, r) => s + r.prompt, 0);
            if (i.cap === undefined) return i.queue.length + i.running.length;
            let n = i.queue.length + i.swapped.length;            // queued + decoding (ScheduledInstance.load)
            for (const r of i.running) if (r.done >= r.target) n++;
            return n;
        };
        const pick = arr => arr.reduce((b, i) => load(i) < load(b) ? i : b, arr[0]);

        function finish(r) {
            r.finish = now; nDone++;
            const nxt = successor.get(r.rid);          // closed-loop session: the next turn after the think time
            if (nxt !== undefined) push(now + nxt.think, () => { nxt.arrival = now; arrive(nxt); });
            if (nDone + nRej === reqs.length) stop = true;
        }
        function submit(inst, r) {
            inst.queue.push(r);
            if (!inst.active && !inst.wake) { inst.wake = true; push(now, () => { inst.wake = false; loop(inst); }); }
        }
        function step(inst, cost, batch, then) {
            inst.active = true;
            push(now + cost.time, () => {
                inst.busy += cost.time; inst.steps++; inst.flops += cost.flops; inst.bytes += cost.bytes;
                inst.batchSum += batch; inst.ec += cost.ec; inst.em += cost.em;
                if (cost.oj || cost.bound === 'optical') {      // account_optical in sim.py
                    inst.oj += cost.oj; inst.oflops += cost.oflops;
                    const pw = inst.cm.idleW + inst.cm.optStaticW + (cost.ec + cost.em + cost.oj) / cost.time; if (pw > inst.peakW) inst.peakW = pw;
                    if (cost.bound === 'power') inst.powerBound += cost.time;
                    else if (cost.bound === 'optical') inst.opticalBound += cost.time;
                } else {
                    const pw = inst.cm.idleW + (cost.ec + cost.em) / cost.time; if (pw > inst.peakW) inst.peakW = pw;
                    if (cost.bound === 'power') inst.powerBound += cost.time;
                }
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
        // CED, replay on decode: new rows run their last w prompt tokens through the whole model in the
        // same step as the running rows' decode, and emit their first token (sim.CedDecodeInstance)
        function cedOnce(inst) {
            let ctx = 0, b = 0; const replay = [];
            for (const r of inst.running) {
                if (r.tokensOut) { ctx += r.prompt + r.tokensOut; b++; }
                else replay.push([r.prompt, Math.min(r.prompt, model.cedW)]);
            }
            step(inst, inst.cm.cedStep(ctx, b, replay), inst.running.length, () => {
                const still = [];
                for (const r of inst.running) {
                    if (r.tokensOut) { r.itls.push(now - r.lastToken); r.tokensOut++; r.lastToken = now; }
                    else firstToken(r);
                    if (r.tokensOut >= r.output) { inst.kvUsed -= kvNeed(r); finish(r); } else still.push(r);
                }
                inst.running = still;
            });
        }
        // ── ScheduledInstance (sim.py): batching policy, KV policy, preemption, prefix cache ──
        const units = (inst, n) => Math.floor((n + inst.blk - 1) / inst.blk);
        const byArrival = (a, b) => a.arrival - b.arrival || a.rid - b.rid;
        const pow2 = o => pow2AtLeast(o);
        function removeRow(rows, r) { const i = rows.indexOf(r); if (i >= 0) rows.splice(i, 1); }
        function fullNeed(inst, r) {
            const p = r.prompt, o = r.output;
            if (inst.paged) return units(inst, p + o) + inst.watermark;
            if (kvPolicy === 'pow2') return p + pow2(o);
            if (kvPolicy === 'max') return Math.max(maxSeqLen, p + o);
            return p + o;
        }
        function admitNeed(inst, r, hit) {
            if (inst.paged) return units(inst, r.target - hit);
            if (kvPolicy === 'pow2') return r.prompt - hit + pow2(r.output);
            if (kvPolicy === 'max') return Math.max(maxSeqLen, r.prompt + r.output) - hit;
            return r.prompt - hit + r.output;
        }
        // PrefixCache: LRU over unpinned leaves, ties by insertion order
        function cacheEvict(c, need) {
            let freed = 0;
            while (freed < need) {
                let best = null;
                for (const n of c.nodes.values())
                    if (n.refs === 0 && n.children === 0 && (best === null || n.last < best.last || (n.last === best.last && n.seq < best.seq))) best = n;
                if (best === null) break;
                c.nodes.delete(best.key); freed += best.units; c.evictedTokens += best.tokens; c.evictedUnits += best.units;
                if (best.parent !== null) best.parent.children--;
            }
            return freed;
        }
        function room(inst, need) {
            const short = inst.kvUsed + need - inst.cap;
            if (short > 0 && inst.cache) inst.kvUsed -= cacheEvict(inst.cache, short);
            return inst.kvUsed + need <= inst.cap;
        }
        function setAlloc(inst, r, u) { inst.kvUsed += u - r.alloc; r.alloc = u; }
        const ref = n => { n.refs++; n.last = now; };
        const unref = n => { n.refs--; n.last = now; };
        function release(inst, r) {
            inst.kvUsed -= r.alloc; r.alloc = 0;
            if (r.nodes.length) { for (const n of r.nodes) unref(n); r.nodes = []; }
            r.covered = 0;
        }
        function lookup(inst, r) {
            if (!inst.cache || !r.prefix.length) return [[], 0];
            const nodes = []; let hit = 0;
            for (const [k] of r.prefix) { const n = inst.cache.nodes.get(k); if (n === undefined) break; nodes.push(n); }
            for (const n of nodes) hit += n.tokens;
            while (nodes.length && hit > r.target - 1) hit -= nodes.pop().tokens;
            return [nodes, hit];
        }
        function tryAdmit(inst) {
            const r = inst.queue[0];
            if (r.target === 0) r.target = r.prompt;
            const [nodes, hit] = lookup(inst, r);
            const need = admitNeed(inst, r, hit);
            for (const n of nodes) ref(n);
            if (!room(inst, need + inst.watermark)) { for (const n of nodes) unref(n); return null; }
            inst.queue.shift();
            r.nodes = nodes; r.covered = hit; r.alloc = 0;
            setAlloc(inst, r, need);
            r.done = hit; r.baseOut = r.tokensOut;
            if (r.prefillStart === null) { r.prefillStart = now; inst.promptTokens += r.prompt; inst.hitTokens += hit; }
            inst.running.push(r); inst.running.sort(byArrival);
            return r;
        }
        function grow(inst) {
            if (!inst.paged) return;
            let i = 0;
            while (i < inst.running.length) {
                const r = inst.running[i];
                if (r.done < r.target) { i++; continue; }
                const need = units(inst, r.prompt + r.tokensOut - r.covered);
                if (need <= r.alloc) { i++; continue; }
                let gone = false;
                while (!room(inst, need - r.alloc)) {
                    const v = inst.running[inst.running.length - 1];
                    preempt(inst, v);
                    if (v === r) { gone = true; break; }
                }
                if (!gone) { setAlloc(inst, r, need); i++; }
            }
        }
        function preempt(inst, r) {
            removeRow(inst.running, r); inst.preemptions++;
            if (inst.swap && r.done >= r.target) {
                const nbytes = r.alloc * inst.blk * model.kvTok;
                const tt = r.alloc * hostLink.lat + nbytes / hostLink.bw;
                inst.swapT += tt; inst.swapTime += tt; inst.swapOut += nbytes;
                inst.swapJ += nbytes * 8 * hostLink.pjBit * 1e-12;
                r.swappedUnits = r.alloc; inst.kvUsed -= r.alloc; r.alloc = 0;
                inst.swapped.push(r); inst.swapped.sort(byArrival);
                return;
            }
            release(inst, r);
            r.target = r.prompt + r.tokensOut; r.done = 0;
            inst.queue.unshift(r);
        }
        function swapIn(inst) {
            while (inst.swapped.length && inst.running.length < maxB) {
                const r = inst.swapped[0];
                if (!room(inst, r.swappedUnits)) break;
                inst.swapped.shift();
                const nbytes = r.swappedUnits * inst.blk * model.kvTok;
                const tt = r.swappedUnits * hostLink.lat + nbytes / hostLink.bw;
                inst.swapT += tt; inst.swapTime += tt; inst.swapIn += nbytes;
                inst.swapJ += nbytes * 8 * hostLink.pjBit * 1e-12;
                inst.kvUsed += r.swappedUnits; r.alloc = r.swappedUnits; r.swappedUnits = 0;
                inst.running.push(r); inst.running.sort(byArrival);
            }
        }
        function privateUnits(inst, r, covered) {
            if (inst.paged) return units(inst, Math.max(0, r.done + r.tokensOut - r.baseOut - covered));
            return r.alloc - (covered - r.covered);
        }
        function insertChain(inst, r) {
            const c = inst.cache;
            if (!c) return;
            for (let k = r.nodes.length; k < r.prefix.length; k++) {
                const [key, tokens] = r.prefix[k];
                const parent = r.nodes.length ? r.nodes[r.nodes.length - 1] : null;
                let n = c.nodes.get(key);
                if (n !== undefined && n.parent === parent) ref(n);
                else {
                    if (n !== undefined) break;
                    const u = units(inst, tokens);
                    if (!room(inst, u + privateUnits(inst, r, r.covered + tokens) - r.alloc)) break;
                    n = { key, tokens, units: u, parent, children: 0, refs: 1, last: now, seq: c.seq++ };
                    if (parent !== null) parent.children++;
                    c.nodes.set(key, n); inst.kvUsed += u;
                }
                setAlloc(inst, r, privateUnits(inst, r, r.covered + tokens));
                r.nodes.push(n); r.covered += tokens;
            }
        }
        function emitSegment(inst, r) {
            const c = inst.cache;
            if (!c || r.emit === null || r.nodes.length !== r.prefix.length) return;
            const tokens = r.prompt - r.covered + r.output, u = units(inst, tokens);
            if (c.nodes.has(r.emit) || !room(inst, u - r.alloc)) return;
            const parent = r.nodes.length ? r.nodes[r.nodes.length - 1] : null;
            const n = { key: r.emit, tokens, units: u, parent, children: 0, refs: 1, last: now, seq: c.seq++ };
            if (parent !== null) parent.children++;
            c.nodes.set(r.emit, n); inst.kvUsed += u; r.nodes.push(n);
        }
        function finishRow(inst, r) { emitSegment(inst, r); release(inst, r); finish(r); }
        function prefillDone(inst, r) {
            insertChain(inst, r);
            if (r.tokensOut) { r.itls.push(now - r.lastToken); r.tokensOut++; r.lastToken = now; }
            else firstToken(r);
            if (r.decodeStart === null && r.output > 1) r.decodeStart = now;
            if (r.tokensOut >= r.output) { removeRow(inst.running, r); finishRow(inst, r); }
        }
        function schedStep(inst, cost, batch, then) {
            if (inst.swapT) { cost = { ...cost, time: cost.time + inst.swapT }; inst.swapT = 0.0; }
            let tok = 0, alloc = 0;
            for (const r of inst.running) { tok += r.done - r.covered + r.tokensOut - r.baseOut; alloc += r.alloc; }
            inst.runArea += cost.time * inst.running.length;
            inst.tokArea += cost.time * tok;
            inst.allocArea += cost.time * (alloc * inst.blk);
            step(inst, cost, batch, then);
        }
        function afterDecode(inst, rows) {
            for (const r of rows) {
                r.itls.push(now - r.lastToken); r.tokensOut++; r.lastToken = now;
                if (r.tokensOut >= r.output) { removeRow(inst.running, r); finishRow(inst, r); }
            }
        }
        function decodeAll(inst) {
            grow(inst);
            if (!inst.running.length) return false;
            const rows = inst.running.slice();
            let ctx = 0; for (const r of rows) ctx += r.prompt + r.tokensOut;
            schedStep(inst, inst.cm.decodeSum(ctx, rows.length), rows.length, () => afterDecode(inst, rows));
            return true;
        }
        function admitPrompts(inst) {
            const batch = []; let tokens = 0;
            while (inst.queue.length && inst.running.length < maxB && !inst.swapped.length) {
                const r = inst.queue[0];
                if (r.target === 0) r.target = r.prompt;
                const tk = r.target - lookup(inst, r)[1];
                if (batch.length && tokens + tk > inst.budget) break;
                if (tryAdmit(inst) === null) break;
                batch.push(r); tokens += r.target - r.done;
            }
            return batch;
        }
        function prefillBatch(inst, batch) {
            const chunks = batch.map(r => [r.done, r.target - r.done, true]);
            let tokens = 0, rec = 0;
            for (let k = 0; k < batch.length; k++) { tokens += chunks[k][1]; if (batch[k].tokensOut) rec += chunks[k][1]; }
            inst.computedTokens += tokens; inst.recomputeTokens += rec;
            schedStep(inst, inst.cm.mixed(0, 0, chunks), batch.length, () => {
                for (const r of batch) { r.done = r.target; prefillDone(inst, r); }
            });
        }
        function chunkedStep(inst) {
            grow(inst);
            const tau = inst.budget;
            const dec = inst.running.filter(r => r.done >= r.target);
            let nt = dec.length, ctx = 0;
            for (const r of dec) ctx += r.prompt + r.tokensOut;
            const part = [];
            for (const r of inst.running)
                if (r.done < r.target && nt < tau) { const c = Math.min(r.target - r.done, tau - nt); part.push([r, c]); nt += c; }
            while (inst.queue.length && nt < tau && inst.running.length < maxB && !inst.swapped.length) {
                const r = tryAdmit(inst);
                if (r === null) break;
                const c = Math.min(r.target - r.done, tau - nt); part.push([r, c]); nt += c;
            }
            if (!dec.length && !part.length) return false;
            const chunks = part.map(([r, c]) => [r.done, c, r.done + c === r.target]);
            let pre = 0, rec = 0;
            for (const [r, c] of part) { pre += c; if (r.tokensOut) rec += c; }
            inst.computedTokens += pre; inst.recomputeTokens += rec;
            schedStep(inst, inst.cm.mixed(ctx, dec.length, chunks), dec.length + part.length, () => {
                afterDecode(inst, dec);
                for (const [r, c] of part) { r.done += c; if (r.done === r.target) prefillDone(inst, r); }
            });
            return true;
        }
        function schedLoop(inst) {
            for (;;) {
                if (inst.swapped.length) swapIn(inst);
                if (policy === 'chunked') { chunkedStep(inst); return; }
                if (policy === 'decode-priority' && inst.running.length) { if (decodeAll(inst)) return; continue; }
                const batch = admitPrompts(inst);
                if (batch.length) { prefillBatch(inst, batch); return; }
                if (inst.running.length) { if (decodeAll(inst)) return; continue; }
                return;
            }
        }
        function loop(inst) {
            if (inst.active) return;
            if (inst.cap !== undefined) { schedLoop(inst); return; }
            if (inst.role === 'prefill') {
                if (!inst.queue.length) return;
                const batch = []; let tok = 0;
                while (inst.queue.length && (!batch.length || tok + inst.queue[0].prompt <= maxPT)) {
                    const r = inst.queue.shift(); batch.push(r); tok += r.prompt; }
                batch.forEach(r => r.prefillStart = now);
                let cost = inst.cm.prefill(batch.map(r => r.prompt));
                if (tr && tr.where === 'endpoint' && !(tr.preset.ratio === 1.0 && !tr.preset.ops)) {
                    let o = 0.0, nb = 0.0;
                    for (const r of batch) { if (r.output <= 1 && !inst.cm.encoderOnly) continue;
                        const hb = model.handoff(r.prompt);
                        o += transitOps(tr, hb, r.prompt, 2.0, true); nb += hb + hb / tr.preset.ratio; }
                    if (nb) cost = inst.cm.compress(cost, o, nb);
                }
                step(inst, cost, batch.length, () => {
                    // CED, replay on decode: the decode instance emits the first token; every request is handed off
                    if (inst.cm.encoderOnly) { for (const r of batch) { r.prefillDone = now; kvRequest(r); } return; }
                    for (const r of batch) { firstToken(r); if (r.output <= 1) finish(r); else kvRequest(r); }
                });
            } else if (inst.role === 'decode') {
                while (inst.queue.length && inst.running.length < maxB && inst.kvUsed + kvNeed(inst.queue[0]) <= inst.cm.kvCap) {
                    const r = inst.queue.shift(); inst.kvUsed += kvNeed(r); r.decodeStart = now; inst.running.push(r); }
                if (inst.running.length) { if (cedOnDecode) cedOnce(inst); else decodeOnce(inst); }
            } else {
                const batch = []; let tok = 0;
                while (inst.queue.length && inst.running.length + batch.length < maxB
                       && (!batch.length || tok + inst.queue[0].prompt <= maxPT)
                       && inst.kvUsed + kvNeed(inst.queue[0]) <= inst.cm.kvCap) {
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
        // (seconds, bytes on the link, link J, in-transit J, transit-bound): Simulation.transfer
        function transfer(r) {
            const nbytes = model.handoff(r.prompt);
            if (!tr) return [link.lat + nbytes / link.bw, nbytes, nbytes * 8 * link.pjBit * 1e-12, 0.0, false];
            const wire = nbytes / tr.preset.ratio;
            if (tr.where === 'endpoint') return [link.lat + wire / link.bw, wire, wire * 8 * link.pjBit * 1e-12, 0.0, false];
            const tWire = wire / link.bw;
            const tOps = transitOps(tr, nbytes, r.prompt, 2.0, false) / (tr.opsPerByte * link.bw);
            return [link.lat + tr.lat + Math.max(tWire, tOps), wire, wire * 8 * link.pjBit * 1e-12,
                    nbytes * 8 * tr.pjBit * 1e-12, tOps > tWire];
        }
        function kvStart(r) {
            ls.inUse++; r.kvStart = now; ls.wait += now - (r.prefillDone ?? r.firstToken);
            const [t, nbytes, joules, transitJ, tb] = transfer(r);
            push(now + t, () => {
                ls.inUse--; ls.busy += t; ls.bytes += nbytes; ls.transfers++; r.kvReady = now;
                ls.energy += joules;
                if (tr) { ls.handoff += model.handoff(r.prompt); ls.transitJ += transitJ; ls.transitBound += tb ? 1 : 0; }
                if (ls.queue.length) kvStart(ls.queue.shift());
                submit(pick(decode), r);
            });
        }
        // arrivals
        function arrive(r) {
            const pool = cfg.mode === 'disagg' ? decode : coloc;
            const tooBig = sched ? fullNeed(front, r) > front.cap : kvNeed(r) > front.cm.kvCap;
            if (tooBig) {
                nRej++; r.rejected = true;
                for (let x = successor.get(r.rid); x !== undefined; x = successor.get(x.rid)) { nRej++; x.rejected = true; }
                if (nDone + nRej === reqs.length) stop = true; return;
            }
            submit(pick(cfg.mode === 'disagg' ? prefill : pool), r);
        }
        for (const r of reqs) if (r.after === null) push(r.arrival, () => arrive(r));
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
        return { cfg, model, reqs, insts, link: ls, linkCh: link.ch, horizon: now, samples, tr, sched,
                 swap: sched && kvPolicy === 'paged' && preemptMode === 'swap' };
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
        const pd = r.prefillDone ?? r.firstToken;       // CED, replay on decode: prefill ends before the first token
        const s = { prefill_queue: r.prefillStart - r.arrival, prefill: pd - r.prefillStart,
                    kv_wait: 0, kv_transfer: 0, decode_queue: 0, decode: 0 };
        if (r.kvStart !== null) { s.kv_wait = r.kvStart - pd; s.kv_transfer = r.kvReady - r.kvStart; }
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
            mfu: i.flops / (H * i.cm.dev.F * i.cm.n),
            mbu: i.bytes / (H * i.cm.dev.B * i.cm.n),
            batch: i.steps ? i.batchSum / i.steps : 0 });
        // energy: static power for the whole run + dynamic work + link (mirror of energy_report)
        const opt = res.insts.some(i => i.cm.dev.transform);
        let st_ = 0, ce = 0, me = 0, os = 0, oc = 0; const perE = {}, optical = {};
        res.insts.forEach(i => { const s_ = i.cm.idleW * H; st_ += s_; ce += i.ec; me += i.em;
            let avg;
            if (opt) { const os_ = i.cm.optStaticW * H; os += os_; oc += i.oj; avg = (s_ + i.ec + i.em + os_ + i.oj) / H;
                       optical[i.name] = { opticalBoundFrac: i.busy ? i.opticalBound / i.busy : 0, opticalFlops: i.oflops,
                                           conversionJ: i.oj, staticJ: os_ }; }
            else avg = (s_ + i.ec + i.em) / H;
            perE[i.name] = { avgW: avg, peakW: i.peakW, powerBoundFrac: i.busy ? i.powerBound / i.busy : 0 }; });
        const outTok = done.reduce((s, r) => s + r.output, 0);
        let totJ = st_ + ce + me + res.link.energy;
        if (opt) totJ = totJ + os + oc;
        if (res.tr) totJ = totJ + res.link.transitJ;
        let swapJ = 0; if (res.swap) { for (const i of res.insts) swapJ += i.swapJ; totJ = totJ + swapJ; }
        const breakdown = { static: st_ / totJ, compute: ce / totJ, memory: me / totJ, link: res.link.energy / totJ };
        if (opt) { breakdown.opticalStatic = os / totJ; breakdown.opticalConversions = oc / totJ; }
        if (res.tr) breakdown.transit = res.link.transitJ / totJ;
        if (res.swap) breakdown.swap = swapJ / totJ;
        const energy = { totalJ: totJ, avgW: totJ / H, jPerTok: totJ / outTok, tokPerJ: outTok / totJ,
            breakdown, perInstance: perE };
        const pools = {};
        for (const i of res.insts) pools[i.role] = `${i.cm.n}x ${i.cm.dev.name}`;
        const lk = res.link;
        const transit = res.tr ? { preset: res.tr.preset.name, where: res.tr.where, ratio: res.tr.preset.ratio,
            handoffGB: lk.handoff / 1e9, transitBoundFrac: lk.transfers ? lk.transitBound / lk.transfers : 0, transitJ: lk.transitJ } : null;
        // brief 20A1 levers (metrics.scheduler_report)
        let scheduler = null;
        if (res.sched) {
            const I = res.insts, i0 = I[0];
            let busy = 0, run = 0, tok = 0, al = 0, comp = 0;
            for (const i of I) { busy += i.busy; run += i.runArea; tok += i.tokArea; al += i.allocArea; comp += i.computedTokens; }
            scheduler = { batchPolicy: cfg.batchPolicy || 'prefill-priority', tokenBudget: i0.budget, kvPolicy: cfg.kvPolicy || 'oracle',
                          kvCapacityUnits: i0.cap, kvUnitTokens: i0.blk, meanRunning: busy ? run / busy : 0.0,
                          kvTokenFrac: al ? tok / al : NaN, computedPrefillTokens: comp };
            if (i0.paged) {
                let pre = 0, rec = 0; for (const i of I) { pre += i.preemptions; rec += i.recomputeTokens; }
                Object.assign(scheduler, { preemption: cfg.preemption || 'recompute', preemptions: pre, recomputeTokens: rec });
                if (i0.swap) {
                    let so = 0, si = 0, ss = 0, sj = 0;
                    for (const i of I) { so += i.swapOut; si += i.swapIn; ss += i.swapTime; sj += i.swapJ; }
                    scheduler.swap = { outGB: so / 1e9, inGB: si / 1e9, seconds: ss, J: sj };
                }
            }
            if (i0.cache) {
                let pt = 0, ht = 0, ev = 0, ce = 0;
                for (const i of I) { pt += i.promptTokens; ht += i.hitTokens; ev += i.cache.evictedTokens;
                                     for (const n of i.cache.nodes.values()) ce += n.tokens; }
                scheduler.prefixCache = { hitRate: pt ? ht / pt : 0.0, hitTokens: ht, promptTokens: pt, evictedTokens: ev, cachedTokensEnd: ce };
            }
        }
        return {
            mode: cfg.mode, completed: done.length, scheduler, measured: steady.length, energy, pools, optical: opt ? optical : null, transit,
            rejected: res.reqs.length - done.length,
            ttft: dist(ttft), tpot: dist(tpot), itl: dist(itl), e2e: dist(e2e),
            tokPerS: done.reduce((s, r) => s + r.output, 0) / H, reqPerS: done.length / H,
            goodput: win > 0 ? met.length / win : NaN, sloAttain: steady.length ? met.length / steady.length : NaN,
            util, eff, stages: st, stageShare: Object.fromEntries(STAGES.map(k => [k, st[k] / e2eMean])),
            hot: { stage: hot, resource: hotRes, util: util[hotRes] },
            kvLink: { GB: res.link.bytes / 1e9, meanWaitMs: 1e3 * res.link.wait / Math.max(1, res.link.transfers) },
            outTokPerS: outTok / H,
            raw: { ttft, itl }, samples: res.samples, horizon: H,
        };
    }
    function run(cfg, rows) {
        return summarise(simulate(cfg, rows));
    }

    // ── PPA (mirror of ppa.py; brief 03's method, illustrative) ────────
    const DIE_MM2 = { 'H100-SXM': 814.0, 'A100-SXM': 826.0, 'Optical-FFT + H100-class': 814.0, 'Optical-FFT + A100-class': 826.0 };
    function murphyYield(a, d0) { const x = (a / 100.0) * d0; if (x === 0) return 1.0; const t = -Math.expm1(-x) / x; return t * t; }
    function diesPerWafer(a, w = 300.0) { const r = w / 2.0;
        const n = Math.PI * r * r / a - Math.PI * w / Math.sqrt(2.0 * a); return Math.max(0, Math.floor(n)); }
    function usdPerGoodDie(a) { const good = diesPerWafer(a, 300.0) * murphyYield(a, 0.1); return good > 0 ? 10000.0 / good : Infinity; }
    function deviceArea(dev) {
        if (!(dev.name in DIE_MM2)) return null;
        let die = DIE_MM2[dev.name], ph = 0.0;
        if (dev.transform) { const ch = Math.ceil(dev.transform.sps / (50.0 * 1e9)); die = die + ch * (0.05 + 0.10); ph = 100.0; }
        let usd = usdPerGoodDie(die); if (ph) usd = usd + usdPerGoodDie(ph);
        return { dieMm2: die, photonicMm2: ph, totalMm2: die + ph, usd };
    }
    function ppa(cfg, m) {
        const roles = cfg.mode === 'disagg' ? [['prefill', cfg.nPrefill], ['decode', cfg.nDecode]] : [['colocated', cfg.nColocated]];
        let mm2 = 0, usd = 0, known = true; const pools = {};
        for (const [role, count] of roles) {
            const key = (role === 'prefill' ? cfg.prefillDevice : role === 'decode' ? cfg.decodeDevice : null) ?? cfg.device;
            const n = (role === 'prefill' ? cfg.prefillDevicesPerInstance : role === 'decode' ? cfg.decodeDevicesPerInstance : null) ?? cfg.devicesPerInstance;
            const a = deviceArea(deviceFor(key, cfg));
            if (!a) { known = false; pools[role] = { device: DEVICES[key].name, mm2: null, usd: null }; continue; }
            pools[role] = { device: DEVICES[key].name, mm2: n * a.totalMm2, usd: n * a.usd };
            mm2 = mm2 + count * n * a.totalMm2; usd = usd + count * n * a.usd;
        }
        return { pools, totalMm2: known ? mm2 : null, totalUsd: known ? usd : null, perfPerW: m.energy.tokPerJ,
                 perfPerMm2: known ? m.outTokPerS / mm2 : null, perfPerKusd: known ? 1e3 * m.outTokPerS / usd : null };
    }
    root.DisaggSim = { MODELS, DEVICES, LINKS, KV_PRESETS, TRANSIT, ENGINE, ENOB_REQUIRED, STAGES, derive, costModel, deviceFor,
                       passes, pjPair, makeWorkload, simulate, summarise, run, percentile, ppa, deviceArea, murphyYield, diesPerWafer };
})(typeof window !== 'undefined' ? window : globalThis);
