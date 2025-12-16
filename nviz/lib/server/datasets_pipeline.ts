import { existsSync, readFileSync, statSync } from "node:fs"
import { join } from "node:path"

import { safeResolve } from "@/lib/server/datasets"
import { datasetManifestPreview } from "@/lib/server/datasets_inspect"

export type FlowStagePresence = {
  raw: boolean
  hydra: boolean
  kept: boolean
  rephrase: boolean
  prep: boolean
}

export type HydraSummaryPreview = {
  total: number
  kept: number
  band: number
  dropped: number
  tau_drop: number | null
  tau_keep: number | null
  top_sources: Array<{ source: string; kept: number; band: number; dropped: number }>
}

export type FlowPipelineInspect = {
  path: string
  is_flow_dir: boolean
  stages: FlowStagePresence
  flow_spec: any | null
  hydra_summary: HydraSummaryPreview | null
  training_shards: {
    path: string
    manifest: ReturnType<typeof datasetManifestPreview>
  } | null
}

function readJsonCapped(absPath: string, capBytes: number): any {
  const st = statSync(absPath)
  if (!st.isFile()) throw new Error("Not a file")
  if (st.size > capBytes) throw new Error(`File too large (${st.size} bytes > ${capBytes} bytes cap)`)
  const txt = readFileSync(absPath, "utf8")
  return JSON.parse(txt)
}

function hydraSummaryPreview(obj: any, topK = 10): HydraSummaryPreview | null {
  if (!obj || typeof obj !== "object") return null
  const total = Number(obj.total ?? 0)
  const kept = Number(obj.kept ?? 0)
  const band = Number(obj.band ?? 0)
  const dropped = Number(obj.dropped ?? 0)
  const tau_drop = Number.isFinite(obj.tau_drop) ? Number(obj.tau_drop) : null
  const tau_keep = Number.isFinite(obj.tau_keep) ? Number(obj.tau_keep) : null
  const per = obj.per_source
  const top_sources: HydraSummaryPreview["top_sources"] = []
  if (per && typeof per === "object") {
    for (const [source, v] of Object.entries(per)) {
      if (!v || typeof v !== "object") continue
      top_sources.push({
        source: String(source),
        kept: Number((v as any).kept ?? 0),
        band: Number((v as any).band ?? 0),
        dropped: Number((v as any).dropped ?? 0),
      })
    }
    top_sources.sort((a, b) => (b.kept - a.kept) || (b.band - a.band) || (b.dropped - a.dropped) || a.source.localeCompare(b.source))
  }
  return {
    total,
    kept,
    band,
    dropped,
    tau_drop,
    tau_keep,
    top_sources: top_sources.slice(0, Math.max(0, Math.min(topK, 50))),
  }
}

export function inspectFlowDir(dirRelPath: string): FlowPipelineInspect {
  const absDir = safeResolve(dirRelPath)
  const st = statSync(absDir)
  if (!st.isDirectory()) throw new Error("Not a directory")

  const flowSpecRel = join(dirRelPath || "", "flow_spec.json")
  const flowSpecAbs = safeResolve(flowSpecRel)
  if (!existsSync(flowSpecAbs)) {
    return {
      path: dirRelPath || "",
      is_flow_dir: false,
      stages: { raw: false, hydra: false, kept: false, rephrase: false, prep: false },
      flow_spec: null,
      hydra_summary: null,
      training_shards: null,
    }
  }

  const rawRel = join(dirRelPath || "", "raw_docs.jsonl")
  const gradesDirRel = join(dirRelPath || "", "hydra_grades")
  const gradesSummaryRel = join(gradesDirRel, "summary.json")
  const keptRel = join(dirRelPath || "", "kept_docs.jsonl")
  const rephrasedRel = join(dirRelPath || "", "rephrased_docs.jsonl")
  const trainingDirRel = join(dirRelPath || "", "training_shards")
  const trainingManifestRel = join(trainingDirRel, "manifest.json")

  const stages: FlowStagePresence = {
    raw: existsSync(safeResolve(rawRel)),
    hydra: existsSync(safeResolve(gradesSummaryRel)),
    kept: existsSync(safeResolve(keptRel)),
    rephrase: existsSync(safeResolve(rephrasedRel)),
    prep: existsSync(safeResolve(trainingManifestRel)),
  }

  let flow_spec: any | null = null
  try {
    flow_spec = readJsonCapped(flowSpecAbs, 4 * 1024 * 1024)
  } catch {
    flow_spec = null
  }

  let hydra_summary: HydraSummaryPreview | null = null
  try {
    const summaryObj = readJsonCapped(safeResolve(gradesSummaryRel), 2 * 1024 * 1024)
    hydra_summary = hydraSummaryPreview(summaryObj)
  } catch {
    hydra_summary = null
  }

  let training_shards: FlowPipelineInspect["training_shards"] = null
  try {
    const manifest = datasetManifestPreview(trainingDirRel, 200)
    training_shards = { path: trainingDirRel, manifest }
  } catch {
    training_shards = null
  }

  return {
    path: dirRelPath || "",
    is_flow_dir: true,
    stages,
    flow_spec,
    hydra_summary,
    training_shards,
  }
}

