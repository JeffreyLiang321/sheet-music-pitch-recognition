// Thin wrapper around Tone.js: schedule the note events on the transport and
// report which ones are sounding so the page can highlight them.
import * as Tone from "tone";

export class Player {
  constructor(onActive, onDone) {
    this.onActive = onActive;
    this.onDone = onDone;
    this.synth = null;
    this.part = null;
  }

  async load(events, bpm) {
    await Tone.start();
    this.stop();
    if (!this.synth) {
      this.synth = new Tone.PolySynth(Tone.Synth, {
        maxPolyphony: 32,
        volume: -8,
        options: { oscillator: { type: "triangle" }, envelope: { attack: 0.01, decay: 0.3, sustain: 0.4, release: 0.6 } },
      }).toDestination();
    }
    const beat = 60 / bpm;
    const active = new Set();
    const notes = events.map((e, i) => ({ time: e.onset * beat, dur: Math.max(0.1, e.duration * beat - 0.05), midi: e.midi, i }));
    const last = notes.reduce((m, n) => Math.max(m, n.time + n.dur), 0);

    this.part = new Tone.Part((time, n) => {
      this.synth.triggerAttackRelease(Tone.Frequency(n.midi, "midi"), n.dur, time);
      Tone.Draw.schedule(() => { active.add(n.i); this.onActive(new Set(active)); }, time);
      Tone.Draw.schedule(() => { active.delete(n.i); this.onActive(new Set(active)); }, time + n.dur);
    }, notes).start(0);

    Tone.Transport.scheduleOnce(() => { this.stop(); this.onDone(); }, last + 0.5);
  }

  play() { Tone.Transport.start(); }
  pause() { Tone.Transport.pause(); }
  stop() {
    Tone.Transport.stop();
    Tone.Transport.cancel();
    if (this.part) { this.part.dispose(); this.part = null; }
    this.onActive(new Set());
  }
  get state() { return Tone.Transport.state; }
}
