export function FigurePanel() {
  return (
    <section id="figure" className="space-y-5">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-amber-800">Figure 1</p>
        <h2 className="mt-1 text-2xl font-semibold tracking-tight">
          All neurons, then the ones SPEC keeps
        </h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-stone-600">
          Panel A is the full four-region raster of one right-lick trial. Panel B is the same
          trial with unselected units faded and selected units marked in gold. Blue dashed
          lines bound the delay (the only context the model may use). Red dotted line marks
          first lick. Reasons for the top units are in the cards below.
        </p>
      </div>
      <figure className="overflow-hidden rounded-2xl border border-stone-200 bg-white shadow-sm">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/fig1_neuron_selection.png"
          alt="Four-region raster with selected delay-predictive neurons highlighted"
          className="h-auto w-full"
        />
      </figure>
      <figure className="overflow-hidden rounded-2xl border border-stone-200 bg-white p-3 shadow-sm">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-500">
          Architecture schematic
        </p>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/fig0_spec_tcnn_schematic.png"
          alt="SPEC-TCNN architecture schematic"
          className="h-auto w-full"
        />
      </figure>
    </section>
  );
}
