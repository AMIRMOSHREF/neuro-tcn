const commands = [
  { title: "Install", cmd: "pip install -r requirements.txt" },
  { title: "Point at your disks", cmd: "edit configs/default.yaml  # Data and Data2 roots" },
  { title: "Or build the demo tree", cmd: "python scripts/prepare_demo.py" },
  { title: "Select + figure + train", cmd: "python scripts/run_pipeline.py --epochs 10" },
  { title: "Train only", cmd: "python scripts/train.py --epochs 20" },
  { title: "Train with in-loop CWT/STFT", cmd: "python scripts/run_pipeline.py --tf --epochs 8" },
];

export function RunBook() {
  return (
    <section id="run" className="space-y-5">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-amber-800">Run</p>
        <h2 className="mt-1 text-2xl font-semibold tracking-tight">Commands</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-stone-600">
          If <code className="rounded bg-stone-200 px-1">C:\\PythonProject\\Rodent\\Data</code> is
          missing, the pipeline falls back to <code className="rounded bg-stone-200 px-1">data/demo</code>.
          Real NPZ files use the same keys as the attached README.
        </p>
      </div>
      <div className="space-y-2">
        {commands.map((c) => (
          <div key={c.title} className="rounded-xl border border-stone-200 bg-[#221910] px-4 py-3 text-[#f6f3ee]">
            <p className="text-[11px] uppercase tracking-wide text-amber-200/80">{c.title}</p>
            <pre className="mt-1 overflow-x-auto text-sm leading-6">
              <code>{c.cmd}</code>
            </pre>
          </div>
        ))}
      </div>
    </section>
  );
}
