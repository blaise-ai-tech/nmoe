import { describe, it, expect } from 'bun:test'
import { NextRequest } from 'next/server'
import { GET as listHandler } from '@/app/api/datasets/route'
import { GET as previewHandler } from '@/app/api/datasets/preview/route'
import { GET as metaHandler } from '@/app/api/datasets/meta/route'
import { GET as manifestHandler } from '@/app/api/datasets/manifest/route'
import { GET as pipelineHandler } from '@/app/api/datasets/pipeline/route'
import { GET as shardStatsHandler } from '@/app/api/datasets/shard/stats/route'
import { GET as shardWindowHandler } from '@/app/api/datasets/shard/window/route'
import { GET as indexHeaderHandler } from '@/app/api/datasets/index/header/route'
import { GET as indexDocHandler } from '@/app/api/datasets/index/doc/route'

import { mkdtempSync, rmSync, mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import os from 'node:os'

function writeNpyUint32(path: string, values: number[]) {
  const magic = Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]) // \x93NUMPY
  const ver = Buffer.from([1, 0])
  const headerText = `{'descr': '<u4', 'fortran_order': False, 'shape': (${values.length},), }`
  const headerBase = Buffer.from(headerText, 'latin1')
  // v1 header: 10-byte preamble; pad to 16-byte alignment and end with newline.
  const preamble = 10
  let pad = 16 - ((preamble + headerBase.length + 1) % 16)
  if (pad === 16) pad = 0
  const header = Buffer.concat([headerBase, Buffer.alloc(pad, ' '), Buffer.from('\n')])
  const headerLen = Buffer.alloc(2)
  headerLen.writeUInt16LE(header.length, 0)
  const data = Buffer.alloc(values.length * 4)
  for (let i = 0; i < values.length; i++) {
    data.writeUInt32LE(values[i] >>> 0, i * 4)
  }
  writeFileSync(path, Buffer.concat([magic, ver, headerLen, header, data]))
}

function writeIdx(path: string, boundaries: Array<[number, number]>) {
  const hdr = Buffer.alloc(32)
  hdr.write('NMOEIDX\u0000', 0, 'latin1')
  hdr.writeBigUInt64LE(1n, 8)
  hdr.writeBigUInt64LE(BigInt(boundaries.length), 16)
  hdr.writeBigUInt64LE(0n, 24)
  const body = Buffer.alloc(boundaries.length * 16)
  for (let i = 0; i < boundaries.length; i++) {
    body.writeBigUInt64LE(BigInt(boundaries[i]![0]), i * 16)
    body.writeBigUInt64LE(BigInt(boundaries[i]![1]), i * 16 + 8)
  }
  writeFileSync(path, Buffer.concat([hdr, body]))
}

describe('/api/datasets contract', () => {
  it('lists dirs and previews files, blocks traversal', async () => {
    const root = mkdtempSync(join(os.tmpdir(), 'nviz-datasets-'))
    process.env.NVIZ_DATASETS_DIR = root
    mkdirSync(join(root, 'sub'), { recursive: true })
    writeFileSync(join(root, 'sub', 'hello.txt'), 'hello\\nworld\\n')

    {
      const url = new URL('http://localhost/api/datasets')
      url.searchParams.set('path', 'sub')
      const req = new NextRequest(url)
      const res = await listHandler(req)
      const body = await res.json() as any
      expect(res.ok).toBe(true)
      expect(body.entries.some((e: any) => e.name === 'hello.txt')).toBe(true)
    }

    {
      const url = new URL('http://localhost/api/datasets/preview')
      url.searchParams.set('path', 'sub/hello.txt')
      const req = new NextRequest(url)
      const res = await previewHandler(req)
      const body = await res.json() as any
      expect(res.ok).toBe(true)
      expect(String(body.text)).toContain('hello')
    }

    {
      const url = new URL('http://localhost/api/datasets')
      url.searchParams.set('path', '../')
      const req = new NextRequest(url)
      const res = await listHandler(req)
      expect(res.status).toBe(400)
    }

    // Inspect endpoints (manifest + shard + index)
    mkdirSync(join(root, 'ds'), { recursive: true })
    const tokens = [1, 2, 3, 999, 4, 5, 999, 6, 999]
    writeNpyUint32(join(root, 'ds', 'shard.npy'), tokens)
    writeIdx(join(root, 'ds', 'shard.idx'), [[0, 4], [4, 7], [7, 9]])
    writeFileSync(join(root, 'ds', 'manifest.json'), JSON.stringify({
      dataset: 'toy',
      version: 'v1',
      tokenizer: 'toy',
      vocab_size: 1000,
      eos_token_id: 999,
      dtype: 'uint32',
      created_at: 'now',
      total_tokens: tokens.length,
      total_documents: 3,
      num_shards: 1,
      shards: [{ path: 'shard.npy', index_path: 'shard.idx', num_tokens: tokens.length, num_documents: 3, checksum: 'x' }],
      source_info: {},
    }))

    {
      const url = new URL('http://localhost/api/datasets/meta')
      url.searchParams.set('path', 'ds')
      const req = new NextRequest(url)
      const res = await metaHandler(req)
      const body = await res.json() as any
      expect(res.ok).toBe(true)
      expect(body.kind).toBe('dir')
    }

    {
      const url = new URL('http://localhost/api/datasets/manifest')
      url.searchParams.set('path', 'ds')
      url.searchParams.set('limit_shards', '10')
      const req = new NextRequest(url)
      const res = await manifestHandler(req)
      const body = await res.json() as any
      expect(res.ok).toBe(true)
      expect(body.found).toBe(true)
      expect(body.eos_token_id).toBe(999)
      expect(body.shards_preview[0].shard_rel).toBe('ds/shard.npy')
    }

    {
      const url = new URL('http://localhost/api/datasets/shard/stats')
      url.searchParams.set('path', 'ds/shard.npy')
      const req = new NextRequest(url)
      const res = await shardStatsHandler(req)
      const body = await res.json() as any
      expect(res.ok).toBe(true)
      expect(body.tokens).toBe(tokens.length)
      expect(body.eos_token_id).toBe(999)
      expect(body.has_index).toBe(true)
    }

    {
      const url = new URL('http://localhost/api/datasets/shard/window')
      url.searchParams.set('path', 'ds/shard.npy')
      url.searchParams.set('offset', '3')
      url.searchParams.set('length', '4')
      const req = new NextRequest(url)
      const res = await shardWindowHandler(req)
      const body = await res.json() as any
      expect(res.ok).toBe(true)
      expect(body.tokens).toEqual([999, 4, 5, 999])
    }

    {
      const url = new URL('http://localhost/api/datasets/index/header')
      url.searchParams.set('path', 'ds/shard.idx')
      const req = new NextRequest(url)
      const res = await indexHeaderHandler(req)
      const body = await res.json() as any
      expect(res.ok).toBe(true)
      expect(body.num_docs).toBe(3)
    }

    {
      const url = new URL('http://localhost/api/datasets/index/doc')
      url.searchParams.set('shard_path', 'ds/shard.npy')
      url.searchParams.set('doc_idx', '1')
      url.searchParams.set('max_tokens', '16')
      const req = new NextRequest(url)
      const res = await indexDocHandler(req)
      const body = await res.json() as any
      expect(res.ok).toBe(true)
      expect(body.start).toBe(4)
      expect(body.end).toBe(7)
      expect(body.tokens).toEqual([4, 5, 999])
    }

    // Flow/pipeline inspection
    mkdirSync(join(root, 'flow', 'hydra_grades'), { recursive: true })
    mkdirSync(join(root, 'flow', 'training_shards'), { recursive: true })
    writeFileSync(join(root, 'flow', 'flow_spec.json'), JSON.stringify({ config_path: 'cfg.toml', config: { run: { name: 'toy' } } }, null, 2))
    writeFileSync(join(root, 'flow', 'raw_docs.jsonl'), '{"id":"a","text":"x"}\n')
    writeFileSync(join(root, 'flow', 'kept_docs.jsonl'), '{"id":"a","text":"x"}\n')
    writeFileSync(join(root, 'flow', 'rephrased_docs.jsonl'), '{"id":"a_v0","text":"y"}\n')
    writeFileSync(join(root, 'flow', 'hydra_grades', 'summary.json'), JSON.stringify({
      total: 10,
      kept: 4,
      band: 6,
      dropped: 0,
      per_source: { fineweb: { kept: 4, band: 6, dropped: 0 } },
      tau_drop: 0.1,
      tau_keep: 1.23,
    }, null, 2))

    writeNpyUint32(join(root, 'flow', 'training_shards', 'shard.npy'), tokens)
    writeIdx(join(root, 'flow', 'training_shards', 'shard.idx'), [[0, 4], [4, 7], [7, 9]])
    writeFileSync(join(root, 'flow', 'training_shards', 'manifest.json'), JSON.stringify({
      dataset: 'toy',
      version: 'v1',
      tokenizer: 'toy',
      vocab_size: 1000,
      eos_token_id: 999,
      dtype: 'uint32',
      created_at: 'now',
      total_tokens: tokens.length,
      total_documents: 3,
      num_shards: 1,
      shards: [{ path: 'shard.npy', index_path: 'shard.idx', num_tokens: tokens.length, num_documents: 3, checksum: 'x' }],
      source_info: {},
    }))

    {
      const url = new URL('http://localhost/api/datasets/pipeline')
      url.searchParams.set('path', 'flow')
      const req = new NextRequest(url)
      const res = await pipelineHandler(req)
      const body = await res.json() as any
      expect(res.ok).toBe(true)
      expect(body.is_flow_dir).toBe(true)
      expect(body.stages.raw).toBe(true)
      expect(body.stages.hydra).toBe(true)
      expect(body.stages.prep).toBe(true)
      expect(body.hydra_summary.total).toBe(10)
      expect(body.training_shards.manifest.found).toBe(true)
      expect(body.training_shards.manifest.dataset).toBe('toy')
    }

    try { rmSync(root, { recursive: true, force: true }) } catch {}
  })
})
