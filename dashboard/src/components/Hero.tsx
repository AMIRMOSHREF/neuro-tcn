type Props = {
  nTrials: number;
  nSelected: number;
  nScored: number;
  laterality: Record<string, number>;
  metrics: Record<string, unknown>;
};

export function Hero({ nTrials, nSelected, nScored, laterality, metrics }: Props) {
  const acc = (metrics as { best_val_acc?: number }).best_val_acc;
  const stats = [
    { label: "Trials (Data + Data2)", value: String(nTrials) },
    { label: "Units scored", value: String(nScored) },
    { label: "Selected ensemble", value: nScored ? `${nSelected} (${Math.round((nSelected / nScored) * 100)}%)` : "—" },
    { label: "Held-out accuracy", value: acc != null ? `${(Number(acc) * 100).toFixed(1)}%` : "run train" },
    {
      label: "Left ALM → Right pref.",
      value:
        laterality.left_ALM_right_pref != null
          ? `${Math.round(Number(laterality.left_ALM_right_pref) * 100)}%`
          : "—",
    },
  ];

  return (
    <header className="border-b border-stone-200 bg-[#221910] text-[#f6f3ee]">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-5 py-10 sm:px-8 sm:py-14">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-amber-200/80">
          Delayed-response licking · ALM + striatum · two corpora
        </p>
        <div className="max-w-3xl space-y-4">
          <h1 className="text-3xl font-semibold leading-tight tracking-tight sm:text-5xl">
            SPEC-TCNN selects the delay-period neurons that forecast lick-time activity
          </h1>
          <p className="text-base leading-7 text-stone-300 sm:text-lg">
            Dilated causal convolutions, neuron/temporal attention, and wavelet/STFT
            features read 1.2 s of delay in left/right ALM and striatum, reconstruct the
            lick-period rasters, and classify Ignore / Left / Right without seeing the go cue.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          {stats.map((s) => (
            <div key={s.label} className="rounded-xl border border-white/10 bg-white/5 px-3 py-3">
              <div className="text-xl font-semibold text-amber-100 sm:text-2xl">{s.value}</div>
              <div className="mt-1 text-[11px] uppercase tracking-wide text-stone-400">{s.label}</div>
            </div>
          ))}
        </div>
        <nav className="flex flex-wrap gap-4 text-sm text-amber-100/90">
          <a href="#figure" className="underline-offset-4 hover:underline">
            Selection figure
          </a>
          <a href="#neurons" className="underline-offset-4 hover:underline">
            Why these units
          </a>
          <a href="#claims" className="underline-offset-4 hover:underline">
            Paper claims
          </a>
          <a href="#methods" className="underline-offset-4 hover:underline">
            Methods
          </a>
          <a href="#run" className="underline-offset-4 hover:underline">
            Commands
          </a>
        </nav>
      </div>
    </header>
  );
}
