#!/usr/bin/env python3
"""hear.py - gate G6 "the voice is heard": check every VO line in the finished film is audible over the bed and understood by ASR.

Usage:
  python3 hear.py film/final.mp4 [--stems film/audio/mix.stems] [--lines film/audio/vo/lines.index.json]
                  [--model small] [--lang en] [--no-asr] [--out film/look-final/hear.json] [--strict]

Needs ffmpeg and numpy. The ASR half needs faster-whisper (`pip install faster-whisper`; the model, ~480 MB for `small`, downloads
to the Hugging Face cache on first use). Without it the masking half still runs and the ASR half prints NOT CHECKED.

Two measurements per VO line (lines come from mix.py's <mix>.stems/manifest.json, text from voice.py split's lines.index.json):
  MASKING  from the stems mix.py writes (vo.wav vs music.wav + sfx.wav, same gains and ducks as the mix): speech-band (300-4000 Hz)
           level of the voice minus the bed, per 100 ms, over the frames where the line is speaking. Prints the line's
           power-averaged ratio, the worst 300 ms and what sits under it (music level, SFX level, the SFX placements that overlap). No unmixing or guessing:
           the stems are exact.
  ASR      faster-whisper on the delivered film's audio (what the viewer hears) and, as a control, on each dry line clip. Word
           error rate against the line's text. Dry clean + mix wrong = the mix hides the line. Both wrong = the voice itself
           is unclear or the word is unusual (a brand name): listen to the clip and judge; regenerate or rephrase if it is the
           voice. No initial prompt is given to the ASR, so it cannot "hear" the script it was told to expect.

G6 FAIL when a line's voice-over-bed is under --min-line dB (default 0: the bed is as loud as the voice across the line), its
worst 300 ms is under --min-worst dB (default -10: a word is buried), or the ASR hears the dry clip but not the mix (WER over
--max-wer 0.2 and at least 0.2 worse than the dry clip). Also FAIL when the stems explain under --min-fit (0.88) of the film's
audio: they are from another mix. WARN at under +8 dB for the line / 0 dB for the worst 300 ms, or when the dry clip itself transcribes wrong.
Calibrated on one real film (Tidepool, 2026-10-27): a wash 25 dB over a word made ASR mishear it; music at -0.1 dB over a line did not.
Writes hear.json (default: next to the film). Exit 0; --strict exits 3 on FAIL; 1 on error; 2 on bad usage.
"""
from __future__ import annotations
import argparse, json, math, os, re, shutil, subprocess, sys

try:
    import numpy as np
except ImportError:
    print("error: hear.py needs numpy (pip install numpy)", file=sys.stderr)
    sys.exit(1)

SR = 16000
HOP = 0.1          # analysis frame, s
BAND = (300.0, 4000.0)
ACTIVE_DB = 20.0   # a frame is "speaking" when the voice is within this many dB of the line's loudest frame


def die(msg, code=1):
    print("error: " + msg, file=sys.stderr)
    sys.exit(code)


def decode(path):
    """Any audio or video file -> 16 kHz mono float32."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
    if r.returncode:
        die("ffmpeg could not decode %s: %s" % (path, r.stderr.decode(errors="replace")[-400:]))
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def band_db(y):
    """Speech-band power per HOP frame, dB."""
    n = int(SR * HOP)
    k = len(y) // n
    if k == 0:
        return np.zeros(0)
    fr = y[:k * n].reshape(k, n) * np.hanning(n)
    spec = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / SR)
    m = (f >= BAND[0]) & (f <= BAND[1])
    return 10 * np.log10(spec[:, m].sum(axis=1) + 1e-12)


def pow_mean_db(x):
    return float(10 * np.log10(np.mean(10 ** (np.asarray(x) / 10)) + 1e-12))


ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    return str(n)


def norm(text):
    text = re.sub(r"\b\d{1,2}\b", lambda m: num_words(int(m.group(0))), text.lower().replace("’", "'"))
    return re.sub(r"[^a-z0-9' ]+", " ", text).replace("'", "").split()


def wer(ref, hyp):
    if not ref:
        return 0.0
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / len(ref)


def clip_dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def find_index(manifest, given):
    if given:
        return given
    for v in manifest.get("vo", []):
        p = os.path.join(os.path.dirname(v["clip"]), "lines.index.json")
        if os.path.isfile(p):
            return p
    return None


def load_asr(model_name):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None, "faster-whisper is not installed (pip install faster-whisper)"
    try:
        return WhisperModel(model_name, device="cpu", compute_type="int8"), None
    except Exception as e:          # model download or load failure
        return None, "could not load whisper model %s: %s" % (model_name, e)


def transcribe(model, y, lang):
    # pass samples, not a path: faster-whisper's own decoder breaks with some PyAV versions
    segs, _ = model.transcribe(y, language=lang, word_timestamps=True, vad_filter=False, condition_on_previous_text=False)
    return [{"w": w.word.strip(), "s": float(w.start), "e": float(w.end), "p": float(w.probability)} for s in segs for w in (s.words or [])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("film")
    ap.add_argument("--stems", help="mix.py stems dir (default: <film>.stems, then audio/mix.stems beside the film, then the newest *.stems under its folder)")
    ap.add_argument("--lines", help="voice.py split lines.index.json (default: beside the VO clips)")
    ap.add_argument("--model", default="small")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--min-line", type=float, default=0.0)
    ap.add_argument("--min-worst", type=float, default=-10.0)
    ap.add_argument("--max-wer", type=float, default=0.2)
    ap.add_argument("--min-fit", type=float, default=0.88, help="stems must explain this share of the film's audio (R2)")
    ap.add_argument("--out")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    if not shutil.which("ffmpeg"):
        die("ffmpeg not found")
    if not os.path.isfile(a.film):
        die("film not found: " + a.film, 2)

    sd = a.stems
    if not sd:
        base = os.path.dirname(os.path.abspath(a.film))
        found = []
        for root, dirs, _ in os.walk(base):
            found += [os.path.join(root, d) for d in dirs if d.endswith(".stems")]
        found.sort(key=lambda d: -os.path.getmtime(d))         # newest mix first
        cands = [os.path.splitext(os.path.abspath(a.film))[0] + ".stems", os.path.join(base, "audio", "mix.stems")] + found
        sd = next((c for c in cands if os.path.isfile(os.path.join(c, "manifest.json"))), None)
    if not sd or not os.path.isfile(os.path.join(sd, "manifest.json")):
        die("no mix.py stems found (need <mix>.stems/manifest.json; re-run mix.py without --no-stems, or pass --stems)", 2)
    man = json.load(open(os.path.join(sd, "manifest.json")))
    if not man.get("vo"):
        print("G6 voice is heard: n/a (no VO in this mix)")
        return 0

    idx = find_index(man, a.lines)
    texts = {}
    if idx:
        for e in json.load(open(idx)):
            texts[os.path.basename(e["clip"].replace("\\", "/"))] = e["text"]

    st = man.get("stems", {})
    vo = decode(os.path.join(sd, st["vo"])) if "vo" in st else None
    bed_parts = {k: decode(os.path.join(sd, st[k])) for k in ("music", "sfx") if k in st}
    if vo is None:
        die("stems dir has no vo.wav", 2)
    n = len(vo)
    bed_parts = {k: np.pad(y, (0, max(0, n - len(y))))[:n] for k, y in bed_parts.items()}
    # Fit each stem's gain to the film's audio. With a linear loudnorm every stem gets the same gain; when loudnorm falls back
    # to dynamic it squashes loud SFX peaks, so the stems are scaled by their fitted gains to grade what the viewer hears.
    # A poor fit means the stems are from another mix (stale) and the masking numbers are not about this film.
    heard_y = decode(a.film)
    m = min(len(heard_y), n)
    cols = [("vo", vo)] + [(k, y) for k, y in bed_parts.items() if np.abs(y).max() > 1e-6]
    A = np.column_stack([c[:m] for _, c in cols])
    g = np.linalg.lstsq(A, heard_y[:m], rcond=None)[0]
    fit_r2 = float(1 - np.sum((heard_y[:m] - A @ g) ** 2) / (np.sum(heard_y[:m] ** 2) + 1e-12))
    gv = max(float(g[0]), 1e-6)
    stale = fit_r2 < a.min_fit or float(g[0]) <= 1e-3
    stem_gain_db = {k: round(20 * math.log10(max(float(gi), 1e-6) / gv), 1) for (k, _), gi in zip(cols, g)}
    for k in bed_parts:
        if k in stem_gain_db:
            bed_parts[k] = bed_parts[k] * (10 ** (stem_gain_db[k] / 20))
    bed = np.zeros(n, dtype=np.float32)
    for y in bed_parts.values():
        bed += y
    v_db, b_db = band_db(vo), band_db(bed)
    part_db = {k: band_db(y) for k, y in bed_parts.items()}
    sfx_spans = [(os.path.basename(s["clip"]), s["t"], s["t"] + clip_dur(s["clip"]), s["gain"]) for s in man.get("sfx", [])]

    lines = []
    for k, v in enumerate(man["vo"]):
        dur = clip_dur(v["clip"])
        i0, i1 = int(v["t"] / HOP), min(len(v_db), int(math.ceil((v["t"] + dur) / HOP)))
        seg = v_db[i0:i1]
        L = {"i": k + 1, "clip": os.path.basename(v["clip"]), "t": v["t"], "text": texts.get(os.path.basename(v["clip"]))}
        if len(seg) == 0 or seg.max() < -80:
            L.update(status="FAIL", why=["no voice found in the vo stem at %.2f s" % v["t"]])
            lines.append(L)
            continue
        act = np.where(seg >= seg.max() - ACTIVE_DB)[0] + i0
        snr = v_db[act] - b_db[act]
        L["speech_s"] = [round(act[0] * HOP, 2), round((act[-1] + 1) * HOP, 2)]
        L["line_db"] = round(pow_mean_db(v_db[act]) - pow_mean_db(b_db[act]), 1)   # speech-to-bed ratio over the spoken frames
        # worst 300 ms: three consecutive frames, power-averaged
        worst, wt = None, None
        for j in range(len(act) - 2):
            if act[j + 2] - act[j] != 2:
                continue
            w = pow_mean_db(v_db[act[j:j + 3]]) - pow_mean_db(b_db[act[j:j + 3]])
            if worst is None or w < worst:
                worst, wt = w, act[j] * HOP
        if worst is None:
            worst, wt = float(snr.min()), act[int(np.argmin(snr))] * HOP
        L["worst_db"], L["worst_at_s"] = round(worst, 1), round(wt, 2)
        j0 = int(round(wt / HOP))
        L["under_worst"] = {k2: round(pow_mean_db(d[j0:j0 + 3]) - pow_mean_db(v_db[j0:j0 + 3]), 1) for k2, d in part_db.items() if pow_mean_db(d[j0:j0 + 3]) > -90}
        L["sfx_overlapping"] = ["%s@%.2f:%g" % (nm, t0, g) for nm, t0, t1, g in sfx_spans if t0 < L["speech_s"][1] and t1 > L["speech_s"][0]]
        lines.append(L)

    asr_note = None
    if not a.no_asr:
        model, asr_note = load_asr(a.model)
        if model:
            heard = transcribe(model, heard_y, a.lang)
            for L in lines:
                if not L.get("text") or "speech_s" not in L:
                    continue
                s0, s1 = L["speech_s"]
                hyp = [w["w"] for w in heard if s0 - 0.3 <= (w["s"] + w["e"]) / 2 <= s1 + 0.3]
                ref = norm(L["text"])
                L["heard_mix"] = " ".join(hyp)
                L["wer_mix"] = round(wer(ref, norm(" ".join(hyp))), 2)
                dry = transcribe(model, decode(next(v["clip"] for v in man["vo"] if os.path.basename(v["clip"]) == L["clip"])), a.lang)
                L["heard_dry"] = " ".join(w["w"] for w in dry)
                L["wer_dry"] = round(wer(ref, norm(L["heard_dry"])), 2)
    elif a.no_asr:
        asr_note = "skipped (--no-asr)"
    if asr_note is None and not any(L.get("text") for L in lines):
        asr_note = "no line text (lines.index.json not found; pass --lines)"

    for L in lines:
        if "status" in L:
            continue
        why, warn = [], []
        if L["line_db"] < a.min_line:
            why.append("over the whole line the bed is as loud as the voice or louder (%+.1f dB, speech band; want >= %g)" % (L["line_db"], a.min_line))
        elif L["line_db"] < 8:
            warn.append("voice only %+.1f dB over the bed across the line (8+ sits clearly on top)" % L["line_db"])
        if L["worst_db"] < a.min_worst:
            why.append("at %.2f s the bed is %.1f dB LOUDER than the voice for 300 ms: a word there is buried" % (L["worst_at_s"], -L["worst_db"]))
        elif L["worst_db"] < 0:
            warn.append("at %.2f s the bed is %.1f dB louder than the voice for 300 ms" % (L["worst_at_s"], -L["worst_db"]))
        if "wer_mix" in L:
            if L["wer_mix"] > a.max_wer and L["wer_dry"] <= a.max_wer and L["wer_mix"] - L["wer_dry"] >= 0.2:
                why.append("ASR understands the dry line but not the mix (WER %.2f vs %.2f dry): the mix hides it" % (L["wer_mix"], L["wer_dry"]))
            elif L["wer_mix"] > a.max_wer and L["wer_dry"] <= a.max_wer:
                warn.append("ASR slips on the mix (WER %.2f vs %.2f dry); listen to %.2f-%.2f s" % (L["wer_mix"], L["wer_dry"], L["speech_s"][0], L["speech_s"][1]))
            elif L["wer_mix"] > a.max_wer:
                warn.append("ASR misreads the dry line too (WER %.2f dry): an unusual word, or the voice itself is unclear; listen to %s and judge" % (L["wer_dry"], L["clip"]))
        L["status"] = "FAIL" if why else ("WARN" if warn else "PASS")
        L["why"] = why + warn

    print("HEAR  %s   stems %s" % (a.film, sd))
    print("  stems fit the film's audio with R2 %.2f; bed scaled to the film: %s" % (fit_r2, ", ".join("%s %+.1f dB" % kv for kv in stem_gain_db.items() if kv[0] != "vo") or "-"))
    if stale:
        print("  STALE STEMS: they explain only %.0f %% of this film's audio (matching stems fit at about 92 %% even after a dynamic"
              " loudnorm), so the levels below are not about this film and G6 cannot pass. Re-run mix.py for the mix in this film"
              " (or pass --stems), re-render, re-run hear.py." % (100 * max(fit_r2, 0)))
    print("  voice over bed is speech-band (300-4000 Hz) level, voice stem minus music + sfx stems, on the frames where the line speaks")
    for L in lines:
        head = "  line %d %-5s %s" % (L["i"], L["status"], L["clip"])
        if "speech_s" in L:
            head += "  %.2f-%.2f s  voice over bed %+.1f dB  worst 300 ms %+.1f dB at %.2f s" % (L["speech_s"][0], L["speech_s"][1], L["line_db"], L["worst_db"], L["worst_at_s"])
        print(head)
        if L.get("text"):
            print("           script: %s" % L["text"])
        if "heard_mix" in L:
            print("           mix   : %s   (WER %.2f)" % (L["heard_mix"], L["wer_mix"]))
            print("           dry   : %s   (WER %.2f)" % (L["heard_dry"], L["wer_dry"]))
        if L["status"] != "PASS" and L.get("under_worst"):
            parts = ", ".join("%s %+.1f dB vs voice" % kv for kv in L["under_worst"].items())
            print("           under the worst 300 ms: %s; sfx overlapping the line: %s" % (parts or "nothing", ", ".join(L["sfx_overlapping"]) or "none"))
        for w in L.get("why", []):
            print("           - " + w)
    if asr_note:
        print("  ASR: NOT CHECKED - %s" % asr_note)
    fails = [L for L in lines if L["status"] == "FAIL"]
    gate = "FAIL" if fails or stale else "PASS"
    print("GATE G6 voice is heard: %s%s" % (gate, " (stale stems)" if stale else "" if not fails else " (lines %s)" % ", ".join(str(L["i"]) for L in fails)))
    if fails and not stale:
        print("  fix in the mix, never by retiming picture: lower or move the SFX under the line (mix.py --sfx gain, or --duck-sfx for beds),"
              " deepen --duck-db, or raise --vo-gain; re-mix, re-render, re-run hear.py")
    out = a.out or os.path.splitext(a.film)[0] + ".hear.json"
    with open(out, "w") as f:
        json.dump({"gate": gate, "stems_fit_r2": round(fit_r2, 3), "stem_gain_db": stem_gain_db, "asr": asr_note or ("faster-whisper " + a.model), "lines": lines, "stems": sd}, f, indent=1)
    print("  -> %s" % out)
    return 3 if (a.strict and fails) else 0


if __name__ == "__main__":
    sys.exit(main())
