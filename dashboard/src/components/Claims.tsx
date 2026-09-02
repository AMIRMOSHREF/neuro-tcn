type Claim = {
  id: string;
  title: string;
  claim: string;
  prediction: string;
  literature: string;
};

export function Claims({ claims }: { claims: Claim[] }) {
  const list = claims.length
    ? claims
    : [
        {
          id: "C1",
          title: "Sparse delay ensemble forecasts lick-period activity",
          claim: "A minority of ALM and striatal units carry a causal predictive code.",
          prediction: "Occluding the selected set raises lick-raster error more than a random set.",
          literature: "Li, Chen, Svoboda; Inagaki et al.",
        },
      ];

  return (
    <section id="claims" className="space-y-5">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-amber-800">
          Paper claims
        </p>
        <h2 className="mt-1 text-2xl font-semibold tracking-tight">
          What you can defend from this analysis
        </h2>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {list.map((c) => (
          <article key={c.id} className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-wide text-amber-800">{c.id}</p>
            <h3 className="mt-1 text-lg font-semibold leading-snug">{c.title}</h3>
            <p className="mt-2 text-sm leading-6 text-stone-700">{c.claim}</p>
            <p className="mt-3 text-sm leading-6 text-stone-600">
              <span className="font-semibold text-stone-800">Test. </span>
              {c.prediction}
            </p>
            <p className="mt-2 text-xs text-stone-500">{c.literature}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
