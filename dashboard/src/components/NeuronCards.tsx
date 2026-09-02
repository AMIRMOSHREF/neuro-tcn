type Neuron = {
  region: string;
  unit_id: number;
  score: number;
  dprime: number;
  delay_lick_coupling: number;
  tf_selectivity: number;
  preferred_class: string;
  neuron_type?: string | null;
  reasons: string;
};

export function NeuronCards({ neurons }: { neurons: Neuron[] }) {
  return (
    <section id="neurons" className="space-y-5">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-amber-800">Selection</p>
        <h2 className="mt-1 text-2xl font-semibold tracking-tight">Why these neurons were kept</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-stone-600">
          Each unit is scored inside its region, then the top 18% are retained. The bullets
          are the criteria that actually fired for that cell — not generic captions.
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {neurons.slice(0, 8).map((n) => (
          <article
            key={`${n.region}-${n.unit_id}`}
            className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="text-base font-semibold">
                  {n.region} · unit {n.unit_id}
                </h3>
                <p className="text-xs text-stone-500">
                  prefers {n.preferred_class}
                  {n.neuron_type ? ` · ${n.neuron_type.replaceAll("_", " ")}` : ""}
                </p>
              </div>
              <span className="rounded-full bg-amber-100 px-2 py-1 text-xs font-semibold text-amber-900">
                score {Number(n.score).toFixed(2)}
              </span>
            </div>
            <dl className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
              <div className="rounded-lg bg-stone-50 py-2">
                <dt className="text-stone-500">d′</dt>
                <dd className="font-semibold">{Number(n.dprime).toFixed(2)}</dd>
              </div>
              <div className="rounded-lg bg-stone-50 py-2">
                <dt className="text-stone-500">delay→lick r</dt>
                <dd className="font-semibold">{Number(n.delay_lick_coupling).toFixed(2)}</dd>
              </div>
              <div className="rounded-lg bg-stone-50 py-2">
                <dt className="text-stone-500">TF sel.</dt>
                <dd className="font-semibold">{Number(n.tf_selectivity).toFixed(2)}</dd>
              </div>
            </dl>
            <ul className="mt-3 space-y-1.5 text-sm leading-5 text-stone-700">
              {String(n.reasons || "")
                .split(" | ")
                .filter(Boolean)
                .slice(0, 4)
                .map((reason) => (
                  <li key={reason} className="flex gap-2">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-700" />
                    <span>{reason}</span>
                  </li>
                ))}
            </ul>
          </article>
        ))}
      </div>
    </section>
  );
}
