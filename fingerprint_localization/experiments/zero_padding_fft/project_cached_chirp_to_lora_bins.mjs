#!/usr/bin/env node
/*
 * Project cached complex chirp CIR profiles into LoRa dechirped FFT bins.
 *
 * This is a dependency-light companion to project_chirp_to_lora_bins.py.  It
 * reads the step6b cache_complex_cir/*.npz files directly through the system
 * unzip command and uses a small in-file radix-2 FFT, so it can run even when
 * the local Python numerical stack is unavailable.
 */

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const DEFAULT_CACHE_DIR = path.join(
  ROOT,
  "v2_output/20260623_from_raw/step6b_chirp_calibrated_clean/cache_complex_cir",
);
const DEFAULT_LORA_INPUT = path.join(
  ROOT,
  "v2_output/20260623_from_raw/data_processing/lora_frequency_s17_54points.csv",
);
const DEFAULT_SUBBIN_INPUT = path.join(
  ROOT,
  "v2_output/20260624_zero_padding_fft_q4_from_trusted_starts/subbin_spectrum_long.csv",
);
const DEFAULT_OUTPUT_DIR = path.join(
  ROOT,
  "v2_output/20260710_chirp_lora_bin_window_scan_projection",
);

const LORA_BW = 125e3;
const SF = 11;
const FFT_SIZE = 1 << SF;
const EPS = 1e-12;

function parseArgs(argv) {
  const args = {
    cacheDir: DEFAULT_CACHE_DIR,
    loraInput: DEFAULT_LORA_INPUT,
    subbinInput: DEFAULT_SUBBIN_INPUT,
    outputDir: DEFAULT_OUTPUT_DIR,
    binRadius: 8,
    q: 1,
    maxSegments: 40,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) {
        throw new Error(`Missing value after ${arg}`);
      }
      return argv[i];
    };
    if (arg === "--cache-dir") args.cacheDir = next();
    else if (arg === "--lora-input") args.loraInput = next();
    else if (arg === "--subbin-input") args.subbinInput = next();
    else if (arg === "--output-dir") args.outputDir = next();
    else if (arg === "--bin-radius") args.binRadius = Number.parseInt(next(), 10);
    else if (arg === "--q") args.q = Number.parseInt(next(), 10);
    else if (arg === "--max-segments") args.maxSegments = Number.parseInt(next(), 10);
    else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!Number.isInteger(args.binRadius) || args.binRadius < 1) {
    throw new Error("--bin-radius must be a positive integer");
  }
  if (!Number.isInteger(args.maxSegments) || args.maxSegments < 1) {
    throw new Error("--max-segments must be a positive integer");
  }
  if (!Number.isInteger(args.q) || args.q < 1) {
    throw new Error("--q must be a positive integer");
  }
  return args;
}

function formatOffset(offset) {
  const rounded = Math.round(offset * 1e6) / 1e6;
  let text = Number.isInteger(rounded) ? `${rounded}` : rounded.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  if (Object.is(rounded, -0)) text = "0";
  return `${rounded >= 0 ? "+" : ""}${text}`;
}

function readZipMember(zipPath, memberName) {
  return execFileSync("unzip", ["-p", zipPath, memberName], {
    maxBuffer: 1024 * 1024 * 64,
  });
}

function parseNpy(buffer) {
  if (buffer.toString("latin1", 0, 6) !== "\x93NUMPY") {
    throw new Error("Invalid .npy magic");
  }
  const major = buffer[6];
  let headerLen;
  let offset;
  if (major === 1) {
    headerLen = buffer.readUInt16LE(8);
    offset = 10;
  } else if (major === 2 || major === 3) {
    headerLen = buffer.readUInt32LE(8);
    offset = 12;
  } else {
    throw new Error(`Unsupported .npy version: ${major}`);
  }
  const header = buffer.toString("latin1", offset, offset + headerLen);
  const descr = /'descr': '([^']+)'/.exec(header)?.[1];
  const fortranOrder = /'fortran_order': (True|False)/.exec(header)?.[1] === "True";
  const shapeText = /'shape': \(([^)]*)\)/.exec(header)?.[1] ?? "";
  const shape = shapeText
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number.parseInt(part, 10));
  if (!descr) {
    throw new Error(`Could not parse .npy header: ${header}`);
  }
  if (fortranOrder) {
    throw new Error("Fortran-order .npy arrays are not supported");
  }
  const dataOffset = offset + headerLen;
  const count = shape.length ? shape.reduce((acc, value) => acc * value, 1) : 1;

  if (descr === "<f8") {
    const values = new Float64Array(count);
    for (let i = 0; i < count; i += 1) {
      values[i] = buffer.readDoubleLE(dataOffset + i * 8);
    }
    return { descr, shape, values };
  }
  if (descr === "<i8") {
    const values = new Array(count);
    for (let i = 0; i < count; i += 1) {
      values[i] = Number(buffer.readBigInt64LE(dataOffset + i * 8));
    }
    return { descr, shape, values };
  }
  if (descr === "<c8") {
    const real = new Float64Array(count);
    const imag = new Float64Array(count);
    for (let i = 0; i < count; i += 1) {
      real[i] = buffer.readFloatLE(dataOffset + i * 8);
      imag[i] = buffer.readFloatLE(dataOffset + i * 8 + 4);
    }
    return { descr, shape, real, imag };
  }
  throw new Error(`Unsupported dtype: ${descr}`);
}

function loadNpyFromZip(zipPath, memberName) {
  return parseNpy(readZipMember(zipPath, memberName));
}

function csvParse(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  if (!rows.length) return [];
  const header = rows[0];
  return rows.slice(1).map((values) => Object.fromEntries(header.map((key, idx) => [key, values[idx] ?? ""])));
}

function csvEscape(value) {
  const text = value === null || value === undefined ? "" : String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function writeCsv(filePath, rows, fieldnames) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const lines = [fieldnames.map(csvEscape).join(",")];
  for (const row of rows) {
    lines.push(fieldnames.map((field) => csvEscape(row[field])).join(","));
  }
  fs.writeFileSync(filePath, `${lines.join("\n")}\n`, "utf8");
}

function fft(real, imag, inverse = false) {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      const tr = real[i];
      const ti = imag[i];
      real[i] = real[j];
      imag[i] = imag[j];
      real[j] = tr;
      imag[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const angle = (inverse ? 2.0 : -2.0) * Math.PI / len;
    const wLenR = Math.cos(angle);
    const wLenI = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wr = 1.0;
      let wi = 0.0;
      const half = len >> 1;
      for (let j = 0; j < half; j += 1) {
        const uR = real[i + j];
        const uI = imag[i + j];
        const vR = real[i + j + half] * wr - imag[i + j + half] * wi;
        const vI = real[i + j + half] * wi + imag[i + j + half] * wr;
        real[i + j] = uR + vR;
        imag[i + j] = uI + vI;
        real[i + j + half] = uR - vR;
        imag[i + j + half] = uI - vI;
        const nextWr = wr * wLenR - wi * wLenI;
        wi = wr * wLenI + wi * wLenR;
        wr = nextWr;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i += 1) {
      real[i] /= n;
      imag[i] /= n;
    }
  }
}

function buildLoRaWaveforms() {
  const upR = new Float64Array(FFT_SIZE);
  const upI = new Float64Array(FFT_SIZE);
  const downR = new Float64Array(FFT_SIZE);
  const downI = new Float64Array(FFT_SIZE);
  for (let n = 0; n < FFT_SIZE; n += 1) {
    const phase = Math.PI * n * (n / FFT_SIZE - 1.0);
    upR[n] = Math.cos(phase);
    upI[n] = Math.sin(phase);
    downR[n] = upR[n];
    downI[n] = -upI[n];
  }
  const txR = new Float64Array(upR);
  const txI = new Float64Array(upI);
  fft(txR, txI, false);
  return { txR, txI, downR, downI };
}

function buildFrequencyDelayTable(delaysUs) {
  const tapCount = delaysUs.length;
  const tableR = new Float64Array(FFT_SIZE * tapCount);
  const tableI = new Float64Array(FFT_SIZE * tapCount);
  for (let k = 0; k < FFT_SIZE; k += 1) {
    const fftFreqBin = k < FFT_SIZE / 2 ? k : k - FFT_SIZE;
    const freq = fftFreqBin * LORA_BW / FFT_SIZE;
    for (let t = 0; t < tapCount; t += 1) {
      const angle = -2.0 * Math.PI * freq * delaysUs[t] * 1e-6;
      const idx = k * tapCount + t;
      tableR[idx] = Math.cos(angle);
      tableI[idx] = Math.sin(angle);
    }
  }
  return { tableR, tableI, tapCount };
}

function projectProfile(profileR, profileI, profileOffset, tapCount, table, waveforms, offsetZps, q) {
  let main = -1;
  let maxAbs = -1.0;
  for (let t = 0; t < tapCount; t += 1) {
    const r = profileR[profileOffset + t];
    const im = profileI[profileOffset + t];
    if (!Number.isFinite(r) || !Number.isFinite(im)) return null;
    const abs2 = r * r + im * im;
    if (abs2 > maxAbs) {
      maxAbs = abs2;
      main = t;
    }
  }
  if (main < 0 || maxAbs <= EPS) return null;

  const mainR = profileR[profileOffset + main];
  const mainI = profileI[profileOffset + main];
  const mainAbs = Math.sqrt(mainR * mainR + mainI * mainI) + EPS;
  const phaseR = mainR / mainAbs;
  const phaseI = -mainI / mainAbs;
  const scale = Math.sqrt(maxAbs) + EPS;

  const cirR = new Float64Array(tapCount);
  const cirI = new Float64Array(tapCount);
  for (let t = 0; t < tapCount; t += 1) {
    const r = profileR[profileOffset + t];
    const im = profileI[profileOffset + t];
    cirR[t] = (r * phaseR - im * phaseI) / scale;
    cirI[t] = (r * phaseI + im * phaseR) / scale;
  }

  const rxR = new Float64Array(FFT_SIZE);
  const rxI = new Float64Array(FFT_SIZE);
  for (let k = 0; k < FFT_SIZE; k += 1) {
    let hR = 0.0;
    let hI = 0.0;
    const rowOffset = k * tapCount;
    for (let t = 0; t < tapCount; t += 1) {
      const er = table.tableR[rowOffset + t];
      const ei = table.tableI[rowOffset + t];
      const cr = cirR[t];
      const ci = cirI[t];
      hR += cr * er - ci * ei;
      hI += cr * ei + ci * er;
    }
    rxR[k] = waveforms.txR[k] * hR - waveforms.txI[k] * hI;
    rxI[k] = waveforms.txR[k] * hI + waveforms.txI[k] * hR;
  }

  fft(rxR, rxI, true);
  for (let n = 0; n < FFT_SIZE; n += 1) {
    const r = rxR[n];
    const im = rxI[n];
    const dr = waveforms.downR[n];
    const di = waveforms.downI[n];
    rxR[n] = r * dr - im * di;
    rxI[n] = r * di + im * dr;
  }

  const intSpecR = new Float64Array(rxR);
  const intSpecI = new Float64Array(rxI);
  fft(intSpecR, intSpecI, false);

  let peak = 0;
  let peakAbs = -1.0;
  for (let k = 0; k < FFT_SIZE; k += 1) {
    const abs2 = intSpecR[k] * intSpecR[k] + intSpecI[k] * intSpecI[k];
    if (abs2 > peakAbs) {
      peakAbs = abs2;
      peak = k;
    }
  }

  let specR = intSpecR;
  let specI = intSpecI;
  let specSize = FFT_SIZE;
  let center = peak;
  if (q > 1) {
    specSize = FFT_SIZE * q;
    specR = new Float64Array(specSize);
    specI = new Float64Array(specSize);
    specR.set(rxR);
    specI.set(rxI);
    fft(specR, specI, false);
    center = peak * q;
  }

  const bins = [];
  for (const offZp of offsetZps) {
    const idx = (center + offZp + specSize) % specSize;
    const real = specR[idx];
    const imag = specI[idx];
    bins.push({
      offset: offZp / q,
      offset_zp: offZp,
      real,
      imag,
      mag: Math.sqrt(real * real + imag * imag),
      phase: Math.atan2(imag, real),
    });
  }
  return bins;
}

function relativeDbToCenterFromBins(bins) {
  const center = bins.find((bin) => bin.offset === 0);
  const ref = (center?.mag ?? 0.0) + EPS;
  return bins.map((bin) => 20.0 * Math.log10((bin.mag + EPS) / ref));
}

function relativeDbToPeakFromBins(bins) {
  const ref = Math.max(...bins.map((bin) => bin.mag)) + EPS;
  return bins.map((bin) => 20.0 * Math.log10((bin.mag + EPS) / ref));
}

function normalizedMagFromBins(bins) {
  const ref = Math.max(...bins.map((bin) => bin.mag)) + EPS;
  return bins.map((bin) => bin.mag / ref);
}

function normalizedComplexFromBins(bins) {
  const center = bins.find((bin) => bin.offset === 0);
  const centerMag = (center?.mag ?? 0.0) + EPS;
  const phaseR = (center?.real ?? 1.0) / centerMag;
  const phaseI = -(center?.imag ?? 0.0) / centerMag;
  const peakMag = Math.max(...bins.map((bin) => bin.mag)) + EPS;
  return bins.map((bin) => ({
    real: (bin.real * phaseR - bin.imag * phaseI) / peakMag,
    imag: (bin.real * phaseI + bin.imag * phaseR) / peakMag,
  }));
}

function mean(values) {
  return values.reduce((acc, value) => acc + value, 0.0) / values.length;
}

function summarizeSynth(segmentRows, binOffsets) {
  const groups = new Map();
  for (const row of segmentRows) {
    const key = `${row.corridor_id}_${row.location_id}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const rows = [];
  for (const [key, group] of [...groups.entries()].sort()) {
    const [corridorId, locationId] = key.split("_").map((value) => Number.parseInt(value, 10));
    const out = {
      corridor_id: corridorId,
      location_id: locationId,
      chirp_segment_count: group.length,
      chirp_corr_score_mean: mean(group.map((row) => Number(row.corr_score))),
    };
    for (const off of binOffsets) {
      const tag = formatOffset(off);
      out[`synth_mag_bin_${tag}_mean`] = mean(group.map((row) => Number(row[`synth_mag_bin_${tag}`])));
      out[`synth_rel_db_bin_${tag}_mean`] = mean(group.map((row) => Number(row[`synth_rel_db_bin_${tag}`])));
      out[`synth_rel_peak_db_bin_${tag}_mean`] = mean(group.map((row) => Number(row[`synth_rel_peak_db_bin_${tag}`])));
      out[`synth_mag_norm_bin_${tag}_mean`] = mean(group.map((row) => Number(row[`synth_mag_norm_bin_${tag}`])));
      out[`synth_real_norm_bin_${tag}_mean`] = mean(group.map((row) => Number(row[`synth_real_norm_bin_${tag}`])));
      out[`synth_imag_norm_bin_${tag}_mean`] = mean(group.map((row) => Number(row[`synth_imag_norm_bin_${tag}`])));
      out[`synth_phase_bin_${tag}_mean`] = mean(group.map((row) => Number(row[`synth_phase_bin_${tag}`])));
    }
    rows.push(out);
  }
  return rows.sort((a, b) => a.corridor_id - b.corridor_id || a.location_id - b.location_id);
}

function summarizeMeasuredInteger(loraRows, binOffsets) {
  const groups = new Map();
  for (const row of loraRows) {
    const corridorId = Number.parseInt(row.corridor_id, 10);
    const locationId = Number.parseInt(row.location_id || row.position_id, 10);
    const key = `${corridorId}_${locationId}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const rows = [];
  for (const [key, group] of [...groups.entries()].sort()) {
    const [corridorId, locationId] = key.split("_").map((value) => Number.parseInt(value, 10));
    const out = {
      corridor_id: corridorId,
      location_id: locationId,
      lora_packet_count: group.length,
      lora_detect_score_db_mean: mean(group.map((row) => Number(row.detect_score_db))),
    };
    const mags = [];
    for (const off of binOffsets) {
      const tag = formatOffset(off);
      const col = `preamble_fft_mag_bin_${tag}`;
      const value = mean(group.map((row) => Number(row[col])));
      out[`meas_mag_bin_${tag}_mean`] = value;
      mags.push({ offset: off, mag: value });
    }
    const center = mags.find((item) => item.offset === 0).mag + EPS;
    const peak = Math.max(...mags.map((item) => item.mag)) + EPS;
    for (const item of mags) {
      const tag = formatOffset(item.offset);
      out[`meas_rel_db_bin_${tag}_mean`] = 20.0 * Math.log10((item.mag + EPS) / center);
      out[`meas_rel_peak_db_bin_${tag}_mean`] = 20.0 * Math.log10((item.mag + EPS) / peak);
      out[`meas_mag_norm_bin_${tag}_mean`] = item.mag / peak;
    }
    rows.push(out);
  }
  return rows.sort((a, b) => a.corridor_id - b.corridor_id || a.location_id - b.location_id);
}

function summarizeMeasuredSubbin(subbinRows, binOffsets, q) {
  const wanted = new Map(binOffsets.map((offset) => [formatOffset(offset), offset]));
  const groups = new Map();
  for (const row of subbinRows) {
    if (Number.parseInt(row.q, 10) !== q) continue;
    const offset = Number(row.subbin_offset);
    const tag = formatOffset(offset);
    if (!wanted.has(tag)) continue;
    const corridorId = Number.parseInt(row.corridor_id, 10);
    const locationId = Number.parseInt(row.position_id, 10);
    const key = `${corridorId}_${locationId}`;
    if (!groups.has(key)) {
      groups.set(key, {
        corridorId,
        locationId,
        packetIds: new Set(),
        detectScores: [],
        byOffset: new Map(),
      });
    }
    const group = groups.get(key);
    group.packetIds.add(`${row.file_name}:${row.packet_index}`);
    group.detectScores.push(Number(row.detect_score_db));
    if (!group.byOffset.has(tag)) {
      group.byOffset.set(tag, {
        magRaw: [],
        magNorm: [],
        relPeakDb: [],
        realNorm: [],
        imagNorm: [],
      });
    }
    const item = group.byOffset.get(tag);
    item.magRaw.push(Number(row.mag_raw));
    item.magNorm.push(Number(row.mag_norm));
    item.relPeakDb.push(Number(row.mag_db_rel_peak));
    item.realNorm.push(Number(row.real_norm));
    item.imagNorm.push(Number(row.imag_norm));
  }

  const rows = [];
  for (const group of [...groups.values()].sort((a, b) => a.corridorId - b.corridorId || a.locationId - b.locationId)) {
    if (binOffsets.some((offset) => !group.byOffset.has(formatOffset(offset)))) {
      continue;
    }
    const out = {
      corridor_id: group.corridorId,
      location_id: group.locationId,
      lora_packet_count: group.packetIds.size,
      lora_detect_score_db_mean: mean(group.detectScores),
    };
    for (const offset of binOffsets) {
      const tag = formatOffset(offset);
      const item = group.byOffset.get(tag);
      out[`meas_mag_bin_${tag}_mean`] = mean(item.magRaw);
      out[`meas_mag_norm_bin_${tag}_mean`] = mean(item.magNorm);
      out[`meas_rel_peak_db_bin_${tag}_mean`] = mean(item.relPeakDb);
      out[`meas_rel_db_bin_${tag}_mean`] = mean(item.relPeakDb);
      out[`meas_real_norm_bin_${tag}_mean`] = mean(item.realNorm);
      out[`meas_imag_norm_bin_${tag}_mean`] = mean(item.imagNorm);
    }
    rows.push(out);
  }
  return rows;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const offsetZps = Array.from({ length: args.binRadius * args.q * 2 + 1 }, (_, idx) => idx - args.binRadius * args.q);
  const binOffsets = offsetZps.map((offsetZp) => offsetZp / args.q);
  const cacheFiles = fs
    .readdirSync(args.cacheDir)
    .filter((name) => /^location_\d+\.npz$/.test(name))
    .map((name) => path.join(args.cacheDir, name))
    .sort();
  if (!cacheFiles.length) {
    throw new Error(`No cache files found in ${args.cacheDir}`);
  }

  const firstDelays = loadNpyFromZip(cacheFiles[0], "delays_us.npy").values;
  const table = buildFrequencyDelayTable(firstDelays);
  const waveforms = buildLoRaWaveforms();
  const segmentRows = [];
  const longRows = [];

  for (const cachePath of cacheFiles) {
    const delays = loadNpyFromZip(cachePath, "delays_us.npy").values;
    if (delays.length !== firstDelays.length) {
      throw new Error(`Unexpected delay-axis length in ${cachePath}`);
    }
    for (let i = 0; i < delays.length; i += 1) {
      if (Math.abs(delays[i] - firstDelays[i]) > 1e-9) {
        throw new Error(`Delay axis mismatch in ${cachePath}`);
      }
    }
    const profiles = loadNpyFromZip(cachePath, "profiles.npy");
    const scores = loadNpyFromZip(cachePath, "corr_scores.npy").values;
    const corridorId = loadNpyFromZip(cachePath, "corridor_id.npy").values[0];
    const locationId = loadNpyFromZip(cachePath, "location_id.npy").values[0];
    const rows = profiles.shape[0];
    const cols = profiles.shape[1];
    let emitted = 0;
    for (let segment = 0; segment < rows && emitted < args.maxSegments; segment += 1) {
      const bins = projectProfile(
        profiles.real,
        profiles.imag,
        segment * cols,
        cols,
        table,
        waveforms,
        offsetZps,
        args.q,
      );
      if (!bins) continue;
      emitted += 1;
      const relDb = relativeDbToCenterFromBins(bins);
      const relPeakDb = relativeDbToPeakFromBins(bins);
      const magNorm = normalizedMagFromBins(bins);
      const normComplex = normalizedComplexFromBins(bins);
      const row = {
        file_name: path.basename(cachePath),
        corridor_id: corridorId,
        location_id: locationId,
        segment,
        corr_score: scores[segment],
      };
      for (let idx = 0; idx < bins.length; idx += 1) {
        const tag = formatOffset(bins[idx].offset);
        row[`synth_mag_bin_${tag}`] = bins[idx].mag;
        row[`synth_rel_db_bin_${tag}`] = relDb[idx];
        row[`synth_rel_peak_db_bin_${tag}`] = relPeakDb[idx];
        row[`synth_mag_norm_bin_${tag}`] = magNorm[idx];
        row[`synth_real_norm_bin_${tag}`] = normComplex[idx].real;
        row[`synth_imag_norm_bin_${tag}`] = normComplex[idx].imag;
        row[`synth_phase_bin_${tag}`] = bins[idx].phase;
        longRows.push({
          source: "chirp_synth_cached",
          file_name: path.basename(cachePath),
          corridor_id: corridorId,
          location_id: locationId,
          segment,
          bin_offset: bins[idx].offset,
          k_offset_zp: bins[idx].offset_zp,
          mag: bins[idx].mag,
          rel_db_to_center: relDb[idx],
          rel_db_to_peak: relPeakDb[idx],
          mag_norm: magNorm[idx],
          real_norm: normComplex[idx].real,
          imag_norm: normComplex[idx].imag,
          phase_rad: bins[idx].phase,
        });
      }
      segmentRows.push(row);
    }
    console.log(`[project] location ${locationId}: ${emitted} profiles`);
  }

  const synthSummary = summarizeSynth(segmentRows, binOffsets);
  const measuredInput = args.q === 1 ? args.loraInput : args.subbinInput;
  const measuredRows = csvParse(fs.readFileSync(measuredInput, "utf8"));
  const measuredSummary = args.q === 1
    ? summarizeMeasuredInteger(measuredRows, binOffsets)
    : summarizeMeasuredSubbin(measuredRows, binOffsets, args.q);

  const segmentFields = [
    "file_name",
    "corridor_id",
    "location_id",
    "segment",
    "corr_score",
    ...binOffsets.flatMap((off) => {
      const tag = formatOffset(off);
      return [
        `synth_mag_bin_${tag}`,
        `synth_rel_db_bin_${tag}`,
        `synth_rel_peak_db_bin_${tag}`,
        `synth_mag_norm_bin_${tag}`,
        `synth_real_norm_bin_${tag}`,
        `synth_imag_norm_bin_${tag}`,
        `synth_phase_bin_${tag}`,
      ];
    }),
  ];
  const synthFields = [
    "corridor_id",
    "location_id",
    "chirp_segment_count",
    "chirp_corr_score_mean",
    ...binOffsets.flatMap((off) => {
      const tag = formatOffset(off);
      return [
        `synth_mag_bin_${tag}_mean`,
        `synth_rel_db_bin_${tag}_mean`,
        `synth_rel_peak_db_bin_${tag}_mean`,
        `synth_mag_norm_bin_${tag}_mean`,
        `synth_real_norm_bin_${tag}_mean`,
        `synth_imag_norm_bin_${tag}_mean`,
        `synth_phase_bin_${tag}_mean`,
      ];
    }),
  ];
  const measFields = [
    "corridor_id",
    "location_id",
    "lora_packet_count",
    "lora_detect_score_db_mean",
    ...binOffsets.map((off) => `meas_mag_bin_${formatOffset(off)}_mean`),
    ...binOffsets.map((off) => `meas_rel_db_bin_${formatOffset(off)}_mean`),
    ...binOffsets.map((off) => `meas_rel_peak_db_bin_${formatOffset(off)}_mean`),
    ...binOffsets.map((off) => `meas_mag_norm_bin_${formatOffset(off)}_mean`),
    ...(args.q > 1 ? binOffsets.map((off) => `meas_real_norm_bin_${formatOffset(off)}_mean`) : []),
    ...(args.q > 1 ? binOffsets.map((off) => `meas_imag_norm_bin_${formatOffset(off)}_mean`) : []),
  ];

  fs.mkdirSync(args.outputDir, { recursive: true });
  writeCsv(path.join(args.outputDir, "01_chirp_synth_segment_bins.csv"), segmentRows, segmentFields);
  writeCsv(path.join(args.outputDir, "02_chirp_synth_point_bins.csv"), synthSummary, synthFields);
  writeCsv(path.join(args.outputDir, "03_lora_measured_point_bins.csv"), measuredSummary, measFields);
  writeCsv(
    path.join(args.outputDir, "04_chirp_synth_vs_lora_long.csv"),
    longRows,
    ["source", "file_name", "corridor_id", "location_id", "segment", "bin_offset", "k_offset_zp", "mag", "rel_db_to_center", "rel_db_to_peak", "mag_norm", "real_norm", "imag_norm", "phase_rad"],
  );
  fs.writeFileSync(
    path.join(args.outputDir, "analysis_summary.json"),
    `${JSON.stringify({
      cache_dir: args.cacheDir,
      lora_input: measuredInput,
      bin_radius: args.binRadius,
      q: args.q,
      max_segments_per_location: args.maxSegments,
      chirp_location_count: synthSummary.length,
      measured_lora_location_count: measuredSummary.length,
      common_location_count: synthSummary.filter((row) =>
        measuredSummary.some((mrow) => mrow.corridor_id === row.corridor_id && mrow.location_id === row.location_id),
      ).length,
      delay_axis_us: {
        min: firstDelays[0],
        max: firstDelays[firstDelays.length - 1],
        count: firstDelays.length,
      },
      note: "Projection uses step6b cached complex CIR profiles, not the original raw-correlation +-8 us extraction.",
    }, null, 2)}\n`,
    "utf8",
  );
  fs.writeFileSync(
    path.join(args.outputDir, "README.md"),
    [
      "# Cached chirp-to-LoRa wide-bin projection",
      "",
      "This run projects cached complex chirp CIR profiles into a synthetic LoRa dechirped FFT window.",
      "",
      `- Zero-padding factor q: ${args.q}`,
      `- Bin window: [-${args.binRadius}, +${args.binRadius}] with step ${1 / args.q}`,
      `- Max profiles per chirp location: ${args.maxSegments}`,
      "- Source cache: step6b_chirp_calibrated_clean/cache_complex_cir",
      `- LoRa input: ${measuredInput}`,
      "- Note: the cache delay axis is reused as stored in step6b.",
      "",
    ].join("\n"),
    "utf8",
  );
}

main();
