// Browser port of enterprise_qa/retrieve.py and generate.py (extractive mode, hybrid retrieval, top 5).
// BM25 and TF-IDF scores match the Python exactly. Ties between zero-score passages can break in a different
// order than NumPy's argsort, so on 2 of 52 checked questions a lower-ranked citation or sentence differs.

const TOKEN = /[a-z0-9]+/g;
const STOP = new Set(`a an the and or of to for in on at by from with without into over after
  before about as is are was were be been being this that these those it its
  we you they i he she them our your their what which who whom how when where
  why do does did done can could should would will may might must not no nor
  if then than so such also just only more most other into per`.split(/\s+/));
const WEAK = new Set(`time times page pages document documents company information please tell
  explain describe northstar holdings year years quarter current`.split(/\s+/));
const ABSTAIN_THRESHOLD = 0.18;
const ABSTAIN_MESSAGE =
  "I do not have enough evidence in the documents you can access to answer this question. " +
  "Try a more specific policy, audit, risk, or security topic, or ask an administrator to ingest the relevant file.";

export const ROLE_ACCESS = {
  public: ["admin", "compliance", "auditor", "employee"],
  internal: ["admin", "compliance", "auditor", "employee"],
  confidential: ["admin", "compliance", "auditor"],
  restricted: ["admin", "compliance"],
};

export function tokenize(text) {
  return (text.toLowerCase().match(TOKEN) || []).filter((t) => !STOP.has(t) && t.length > 1);
}

function contentTerms(text) {
  const terms = new Set();
  for (const t of tokenize(text)) {
    if (WEAK.has(t) || (/^\d+$/.test(t) && t.length === 4)) continue;
    terms.add(t);
  }
  return terms;
}

function stems(t) {
  const out = new Set([t]);
  if (t.length <= 4) return out;
  if (t.endsWith("ies")) out.add(t.slice(0, -3) + "y");
  if (t.endsWith("es")) out.add(t.slice(0, -2));
  if (t.endsWith("s")) out.add(t.slice(0, -1));
  if (t.endsWith("ed")) { out.add(t.slice(0, -2)); out.add(t.slice(0, -1)); }
  if (t.endsWith("ing")) out.add(t.slice(0, -3));
  return out;
}

function termInDoc(term, docTerms) {
  const expanded = new Set();
  for (const d of docTerms) for (const s of stems(d)) expanded.add(s);
  for (const v of stems(term)) if (expanded.has(v)) return true;
  for (const d of docTerms) {
    if (Math.min(d.length, term.length) >= 4 && (d.startsWith(term) || term.startsWith(d))) return true;
  }
  return false;
}

function matchedContent(question, text) {
  const docTerms = contentTerms(text);
  return new Set([...contentTerms(question)].filter((t) => termInDoc(t, docTerms)));
}

function minmax(values) {
  const peak = Math.max(...values), floor = Math.min(...values);
  if (!values.length || peak <= 0) return values.map(() => 0);
  if (Math.abs(peak - floor) < 1e-12) return values.map(() => 1);
  return values.map((v) => (v - floor) / (peak - floor));
}

// Descending order by score; ties keep the reversed-index order of np.argsort(...)[::-1].
function argsortDesc(scores) {
  return scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a] || b - a);
}

function ngrams(tokens) {
  const out = [...tokens];
  for (let i = 0; i + 1 < tokens.length; i++) out.push(tokens[i] + " " + tokens[i + 1]);
  return out;
}

function counts(list) {
  const m = new Map();
  for (const x of list) m.set(x, (m.get(x) || 0) + 1);
  return m;
}

// rank_bm25.BM25Okapi (k1=1.5, b=0.75, epsilon=0.25) and sklearn TfidfVectorizer (smooth idf, l2) over one role's chunks.
class Retriever {
  constructor(chunks) {
    this.chunks = chunks;
    const docs = chunks.map((c) => tokenize(c.text));
    this.tf = docs.map(counts);
    this.len = docs.map((d) => d.length);
    this.avgdl = this.len.reduce((a, b) => a + b, 0) / Math.max(1, docs.length);
    const df = new Map();
    for (const tf of this.tf) for (const t of tf.keys()) df.set(t, (df.get(t) || 0) + 1);
    const n = docs.length;
    this.idf = new Map();
    let sum = 0;
    const negatives = [];
    for (const [t, f] of df) {
      const v = Math.log(n - f + 0.5) - Math.log(f + 0.5);
      this.idf.set(t, v);
      sum += v;
      if (v < 0) negatives.push(t);
    }
    const eps = 0.25 * (sum / Math.max(1, df.size));
    for (const t of negatives) this.idf.set(t, eps);

    const grams = docs.map((d) => counts(ngrams(d)));
    const gdf = new Map();
    for (const g of grams) for (const t of g.keys()) gdf.set(t, (gdf.get(t) || 0) + 1);
    this.tidf = new Map([...gdf].map(([t, f]) => [t, Math.log((1 + n) / (1 + f)) + 1]));
    this.vecs = grams.map((g) => this.normalize(g));
  }

  normalize(gramCounts) {
    const v = new Map();
    let norm = 0;
    for (const [t, c] of gramCounts) {
      const idf = this.tidf.get(t);
      if (idf === undefined) continue;
      v.set(t, c * idf);
      norm += (c * idf) ** 2;
    }
    norm = Math.sqrt(norm) || 1;
    for (const [t, x] of v) v.set(t, x / norm);
    return v;
  }

  bm25(query) {
    const q = tokenize(query), k1 = 1.5, b = 0.75;
    return minmax(this.tf.map((tf, i) => {
      let s = 0;
      for (const t of q) {
        const f = tf.get(t) || 0;
        s += (this.idf.get(t) || 0) * (f * (k1 + 1)) / (f + k1 * (1 - b + (b * this.len[i]) / this.avgdl));
      }
      return s;
    }));
  }

  tfidf(query) {
    const qv = this.normalize(counts(ngrams(tokenize(query))));
    return minmax(this.vecs.map((v) => {
      let s = 0;
      for (const [t, x] of qv) s += x * (v.get(t) || 0);
      return s;
    }));
  }

  search(query, topK = 5) {
    if (!this.chunks.length) return [];
    const fused = new Array(this.chunks.length).fill(0);
    for (const scores of [this.bm25(query), this.tfidf(query)]) {
      argsortDesc(scores).forEach((idx, r) => { fused[idx] += 1 / (60 + r + 1); });
    }
    const scores = minmax(fused);
    return argsortDesc(scores).slice(0, topK)
      .map((idx, r) => ({ chunk: this.chunks[idx], score: scores[idx], rank: r + 1 }))
      .filter((p) => p.score > 0);
  }
}

function lexicalOverlap(a, b) {
  const left = counts(tokenize(a)), right = counts(tokenize(b));
  let total = 0, shared = 0;
  for (const c of left.values()) total += c;
  if (!total || !right.size) return 0;
  for (const [t, c] of left) shared += Math.min(c, right.get(t) || 0);
  return shared / Math.max(1, total);
}

function rankSentences(question, passages) {
  const qTerms = contentTerms(question), seen = new Set(), scored = [];
  for (const p of passages) {
    for (let s of p.chunk.text.split(/(?<=[.!?])\s+/)) {
      s = s.trim();
      if (s.length < 40 || seen.has(s.toLowerCase())) continue;
      seen.add(s.toLowerCase());
      const overlap = matchedContent(question, s).size / Math.max(1, qTerms.size);
      if (overlap < 0.34) continue;
      scored.push({ score: 0.75 * overlap + 0.25 * lexicalOverlap(question, s), sentence: s, passage: p });
    }
  }
  return scored.sort((a, b) => b.score - a.score).slice(0, 2);
}

function citations(passages) {
  const seen = new Set(), out = [];
  for (const p of passages) {
    const key = p.chunk.filename + "|" + p.chunk.page;
    if (seen.has(key)) continue;
    seen.add(key);
    let snippet = p.chunk.text.trim();
    if (snippet.length > 280) snippet = snippet.slice(0, 277).trimEnd() + "…";
    out.push({ title: p.chunk.title, page: p.chunk.page, snippet, score: p.score });
  }
  return out;
}

export function answer(question, passages) {
  const qTerms = contentTerms(question);
  const joined = passages.slice(0, 3).map((p) => p.chunk.text).join(" ");
  const matched = matchedContent(question, joined);
  const support = qTerms.size ? matched.size / qTerms.size : 0;
  const coverage = support; // question_coverage() is the same computation over the top 3 passages
  const confidence = Math.round((0.55 * support + 0.45 * coverage) * 1e4) / 1e4;
  const enoughHits = matched.size >= 2 || (qTerms.size <= 1 && matched.size === 1);
  const abstain = (conf) => ({ abstained: true, confidence: conf, answer: ABSTAIN_MESSAGE, citations: citations(passages.slice(0, 2)) });
  if (!passages.length || !enoughHits || support < 0.5 || coverage < 0.45 || confidence < Math.max(ABSTAIN_THRESHOLD, 0.4)) {
    return abstain(confidence);
  }
  const top = rankSentences(question, passages);
  if (!top.length || top[0].score < 0.28) return abstain(Math.min(confidence, 0.35));
  const text = top.map(({ sentence, passage }) =>
    `${/[.?!]$/.test(sentence) ? sentence : sentence + "."} (${passage.chunk.title}, p. ${passage.chunk.page})`).join(" ");
  return { abstained: false, confidence, answer: text, citations: citations(passages) };
}

export function createAssistant(corpus) {
  const cache = new Map();
  return (question, role) => {
    if (!cache.has(role)) cache.set(role, new Retriever(corpus.filter((c) => ROLE_ACCESS[c.classification]?.includes(role))));
    return answer(question, cache.get(role).search(question));
  };
}
