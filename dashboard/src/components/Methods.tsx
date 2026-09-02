const steps = [
  {
    title: "Align both corpora",
    body: "Data uses Session*/Rasters/{Ignore,Left,Right}/trial_*.npz. Data2 uses sub-*/ses-*/NPZ plus audited CSVs. Photostim, early lick, auto-water, and bilateral-lick trials are dropped.",
  },
  {
    title: "Cut delay vs lick",
    body: "Delay is [delay_start, delay_stop] (~1.2 s, 10 ms bins). Lick is 800 ms from first lick, or go-aligned on Ignore trials. The encoder never sees go/lick samples.",
  },
  {
    title: "Four-stream TCNN + DCC",
    body: "Each region is a gated dilated causal conv stack (dilations 1,2,4,8). Receptive field covers the full delay without future leak.",
  },
  {
    title: "Dual attention + TF",
    body: "Neuron attention picks units. Causal temporal attention picks late-delay bins. Morlet CWT and STFT (4–80 Hz) cross-attend into the same space.",
  },
  {
    title: "Two heads, one selected set",
    body: "Head 1 reconstructs lick rasters (Poisson NLL). Head 2 classifies Ignore/Left/Right. L1/entropy on neuron attention keeps the ensemble sparse.",
  },
];

export function Methods() {
  return (
    <section id="methods" className="space-y-5">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-amber-800">Methods</p>
        <h2 className="mt-1 text-2xl font-semibold tracking-tight">
          Selective Predictive Epoch Context
        </h2>
      </div>
      <ol className="grid gap-3">
        {steps.map((s, i) => (
          <li key={s.title} className="flex gap-4 rounded-2xl border border-stone-200 bg-white p-4 shadow-sm">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#221910] text-sm font-semibold text-amber-100">
              {i + 1}
            </span>
            <div>
              <h3 className="font-semibold">{s.title}</h3>
              <p className="mt-1 text-sm leading-6 text-stone-600">{s.body}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
