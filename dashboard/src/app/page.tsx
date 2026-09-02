import { readFile } from "node:fs/promises";
import path from "node:path";
import { Claims } from "@/components/Claims";
import { FigurePanel } from "@/components/FigurePanel";
import { Hero } from "@/components/Hero";
import { Methods } from "@/components/Methods";
import { NeuronCards } from "@/components/NeuronCards";
import { RunBook } from "@/components/RunBook";

export const dynamic = "force-static";

async function loadDashboard() {
  const file = path.join(process.cwd(), "public", "dashboard.json");
  try {
    const raw = await readFile(file, "utf8");
    return JSON.parse(raw);
  } catch {
    return {
      n_trials: 0,
      n_selected: 0,
      n_scored: 0,
      selected: [],
      laterality: {},
      metrics: {},
      claims: [],
    };
  }
}

export default async function Home() {
  const data = await loadDashboard();
  return (
    <div className="flex flex-col">
      <Hero
        nTrials={data.n_trials ?? 0}
        nSelected={data.n_selected ?? 0}
        nScored={data.n_scored ?? 0}
        laterality={data.laterality ?? {}}
        metrics={data.metrics ?? {}}
      />
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-16 px-5 pb-24 pt-10 sm:px-8">
        <FigurePanel />
        <NeuronCards neurons={data.selected ?? []} />
        <Claims claims={data.claims ?? []} />
        <Methods />
        <RunBook />
      </main>
    </div>
  );
}
