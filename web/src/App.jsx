import React, { useEffect, useMemo, useRef, useState } from "react";
import { Player } from "./player";

const CLASS_COLOR = { noteheadBlack: "#d33", noteheadHalf: "#2a7", noteheadWhole: "#27d" };
const CLASS_NAME = { noteheadBlack: "quarter", noteheadHalf: "half", noteheadWhole: "whole" };
const CLEF_NAME = { gClef: "treble", fClef: "bass", cClef: "alto" };
const SAMPLES = [
  { label: "Satie, Gnossienne 1", file: "/samples/satie.png" },
  { label: "Chopin, Etude Op.10 No.1", file: "/samples/chopin.png" },
  { label: "Schumann, String Quartet", file: "/samples/schumann.png" },
];

function keyName(fifths) {
  if (fifths === 0) return "C major / A minor";
  const n = Math.abs(fifths);
  return `${n} ${fifths > 0 ? "sharp" : "flat"}${n > 1 ? "s" : ""}`;
}

export default function App() {
  const [server, setServer] = useState("checking");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [bpm, setBpm] = useState(90);
  const [active, setActive] = useState(new Set());
  const [playing, setPlaying] = useState(false);
  const player = useRef(null);
  const fileInput = useRef(null);

  useEffect(() => {
    player.current = new Player(setActive, () => setPlaying(false));
    // keep pinging until the server answers; on the free tier it may be asleep
    let id;
    const ping = () => fetch("/api/health")
      .then((r) => { if (r.ok) { setServer("ready"); clearInterval(id); } else setServer("waking"); })
      .catch(() => setServer("waking"));
    ping();
    id = setInterval(ping, 5000);
    return () => clearInterval(id);
  }, []);

  async function transcribe(file) {
    setBusy(true); setError(null); setResult(null); player.current.stop(); setPlaying(false);
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("bpm", bpm);
      const r = await fetch("/api/transcribe", { method: "POST", body });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      setResult(await r.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function loadSample(s) {
    const blob = await (await fetch(s.file)).blob();
    transcribe(new File([blob], s.file.split("/").pop(), { type: blob.type }));
  }

  async function togglePlay() {
    const p = player.current;
    if (playing) { p.pause(); setPlaying(false); return; }
    if (p.state !== "paused") await p.load(result.events, bpm);
    p.play(); setPlaying(true);
  }

  function stop() { player.current.stop(); setPlaying(false); }

  const eventsByPage = useMemo(() => {
    if (!result) return [];
    return result.pages.map((_, i) => result.events.map((e, idx) => ({ ...e, idx })).filter((e) => e.page === i));
  }, [result]);

  return (
    <div className="app">
      <header>
        <h1>Sheet to Audio</h1>
        <p>
          Upload a page of piano sheet music (PDF or image). The server finds the staves and noteheads,
          reads each note's pitch from the clef and key signature, and plays the result back.
        </p>
        {server !== "ready" && <p className="notice">Server is waking up, this can take a minute on the free tier.</p>}
        <div className="caveat">
          <strong>What to expect.</strong> Only quarter, half and whole noteheads are recognised:
          eighths and sixteenths (beamed or flagged notes) play as quarters and rests are skipped, so
          rhythmically complex music will sound wrong. Key signatures and accidentals are read, but not
          reliably in dense chords; clef changes and <em>8va</em> marks mid-line are not read at all.
          Works on clean, printed scores, not photos or handwriting.
        </div>
      </header>

      <section className="controls">
        <div className="upload" onClick={() => fileInput.current.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); e.dataTransfer.files[0] && transcribe(e.dataTransfer.files[0]); }}>
          <input ref={fileInput} type="file" accept=".pdf,image/*" hidden
            onChange={(e) => e.target.files[0] && transcribe(e.target.files[0])} />
          {busy ? "Transcribing…" : "Drop a PDF or image here, or click to choose"}
        </div>
        <div className="samples">
          <span>or try a page from the held-out test set:</span>
          {SAMPLES.map((s) => <button key={s.file} disabled={busy} onClick={() => loadSample(s)}>{s.label}</button>)}
        </div>
        <label className="tempo">
          Tempo {bpm} bpm
          <input type="range" min="50" max="160" value={bpm} onChange={(e) => { setBpm(+e.target.value); stop(); }} />
        </label>
      </section>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="result">
          <div className="toolbar">
            <button className="primary" onClick={togglePlay}>{playing ? "Pause" : "Play"}</button>
            <button onClick={stop}>Stop</button>
            <a href={`/api/result/${result.id}/midi`} download>Download MIDI</a>
            <a href={`/api/result/${result.id}/musicxml`} download>Download MusicXML</a>
            <span className="stats">
              {result.stats.pages} page{result.stats.pages > 1 ? "s" : ""} · {result.stats.staffs} staves ·{" "}
              {result.stats.notes} notes · {result.stats.seconds}s
            </span>
          </div>
          <div className="legend">
            <span style={{ color: CLASS_COLOR.noteheadBlack }}>■ quarter</span>
            <span style={{ color: CLASS_COLOR.noteheadHalf }}>■ half</span>
            <span style={{ color: CLASS_COLOR.noteheadWhole }}>■ whole</span>
          </div>
          {result.pages.map((page, pi) => (
            <Page key={pi} page={page} events={eventsByPage[pi]} active={active} />
          ))}
        </section>
      )}

      <footer>
        Started as a CSCI 1470 project at Brown; the pipeline, numbers on held-out pages and remaining
        limitations are described in the project README.
      </footer>
    </div>
  );
}

function Page({ page, events, active }) {
  const s = page.display_scale;
  const w = Math.round(page.width * s);
  const h = Math.round(page.height * s);
  return (
    <div className="page" style={{ width: w }}>
      <img src={page.image} width={w} height={h} alt="score page" />
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
        {page.staffs.map((st, i) => (
          <text key={i} x={Math.max(4, st.left * s - 4)} y={st.top * s - 6} className="stafflabel">
            {CLEF_NAME[st.clef] || st.clef}, {keyName(st.key_fifths)}
          </text>
        ))}
        {events.map((e) => {
          const on = active.has(e.idx);
          return (
            <g key={e.idx}>
              <rect x={e.x0 * s} y={e.y0 * s} width={(e.x1 - e.x0) * s} height={(e.y1 - e.y0) * s}
                rx="2" fill={on ? "#fc0" : "none"} fillOpacity={on ? 0.6 : 0}
                stroke={CLASS_COLOR[e.cls] || "#d33"} strokeWidth={on ? 2.5 : 1.2} />
              <title>{e.name} ({CLASS_NAME[e.cls] || e.cls}), beat {e.onset}</title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
