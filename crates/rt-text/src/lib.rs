//! Native text preparation, calibration, decoding and public APIs for pinned Vela.
//! The pinned Python oracle is retained for conformance tests.
//! Model execution is supplied by the caller; this crate does not load ONNX.
use fancy_regex::Regex;
use rt_core::{check_shape, invalid_shape, ResultContext as Context};
pub use rt_core::{Error, Result, SystemOneError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Bound on [`Frontend`]'s token-id cache (strings).
pub const ID_CACHE_ENTRIES: usize = 4096;

const CJK: &str = "⺀-⿿　-〿぀-ヿ㄀-ㇿ㐀-䶿一-鿿ꀀ-꓏가-힯豈-﫿︰-﹏＀-￯";
const SPACE: &str = r"\s\x1c-\x1f";
const EDGE: &str = "[](){}<>「」『』【】《》〈〉\"'“”‘’«»‹›";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inputs {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
    pub q_index: Vec<[usize; 4]>,
    pub opt_index: Vec<[usize; 3]>,
    pub unit_index: Vec<[usize; 2]>,
    pub ent_index: Vec<[usize; 2]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Logits {
    pub opt_logits: Vec<f32>,
    pub span_logits: Vec<Vec<f32>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Word {
    offset: [usize; 2],
    first: usize,
}
#[derive(Clone)]
struct Part {
    role: String,
    ids: Vec<u32>,
    words: Option<Vec<Word>>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Question {
    qid: String,
    qtype: String,
    over: Value,
    position: usize,
    options: Vec<usize>,
    names: Vec<String>,
    abs: Option<usize>,
    pool: [usize; 2],
}
#[derive(Clone, Serialize, Deserialize)]
struct Record {
    ids: Vec<u32>,
    questions: Vec<Question>,
    names: Vec<String>,
    entities: Vec<usize>,
    word_positions: Option<Vec<usize>>,
    word_offsets: Vec<[usize; 2]>,
    segment_lengths: HashMap<String, usize>,
    protected_cut: i64,
}
#[derive(Clone, Serialize, Deserialize)]
struct Job {
    row: usize,
    windowed: bool,
    global_words: Option<Vec<usize>>,
    record: Record,
}
#[derive(Serialize, Deserialize)]
pub struct Batch {
    pub jobs: Vec<usize>,
    pub inputs: Inputs,
}
#[derive(Serialize, Deserialize)]
pub struct Plan {
    pub batches: Vec<Batch>,
    jobs: Vec<Job>,
    rows: Vec<Value>,
    tokens: Vec<HashMap<String, usize>>,
    full_words: Vec<Vec<[usize; 2]>>,
}
struct BuiltQuestion {
    qid: String,
    qtype: String,
    over: Value,
    text: Vec<u32>,
    options: Vec<(String, Vec<u32>)>,
    abstain: bool,
    length: usize,
}

pub struct Frontend {
    config: Value,
    calibration: Value,
    tokenizer: Tokenizer,
    word: Regex,
    unit: Regex,
    /// Token ids of recently seen strings (questions, options, parts).
    /// Keys come from requests, so it is cleared once it holds
    /// [`ID_CACHE_ENTRIES`] strings.
    cache: Mutex<HashMap<String, Vec<u32>>>,
    pub batch_size: usize,
    pub token_budget: usize,
}
fn text(value: &Value) -> Result<&str> {
    value.as_str().context("expected string")
}
fn array(value: &Value) -> Result<&Vec<Value>> {
    value.as_array().context("expected array")
}
fn truth(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(v) => *v,
        Value::String(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
        Value::Number(v) => v.as_f64() != Some(0.),
    }
}
fn roles(value: &Value) -> Result<Vec<String>> {
    if let Some(s) = value.as_str() {
        Ok(vec![s.into()])
    } else {
        array(value)?.iter().map(|v| Ok(text(v)?.into())).collect()
    }
}
fn names(value: &Value) -> Result<Vec<(String, String)>> {
    Ok(value
        .as_object()
        .context("options/labels must be ordered objects")?
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").into()))
        .collect())
}
fn chars_to_bytes(s: &str) -> Vec<usize> {
    s.char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(s.len()))
        .collect()
}
fn option_text(name: &str, description: &str) -> String {
    if description.is_empty() {
        name.into()
    } else {
        format!("{name}: {description}")
    }
}

impl Frontend {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config = rt_core::read_json(path.join("config.json"))?;
        let calibration = rt_core::read_json(path.join("calibration.json"))?;
        let mut tokenizer_json = rt_core::read_json(path.join("tokenizer.json"))?;
        let markers = config["marker_ids"].as_object().context("marker_ids")?;
        tokenizer_json
            .get_mut("added_tokens")
            .context("tokenizer added_tokens")?
            .as_array_mut()
            .context("added_tokens")?
            .retain(|v| !markers.contains_key(v["content"].as_str().unwrap_or("")));
        let mut tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&tokenizer_json)?)
            .map_err(rt_core::Error::Tokenizer)?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(rt_core::Error::Tokenizer)?;
        // Python \w is letters/numbers/underscore, rather than Rust's broader
        // Unicode word class. Preserve the released CJK ranges and URL lookahead.
        let alnum = format!(r"[\p{{L}}\p{{N}}_&&[^{CJK}]]");
        let word = Regex::new(&format!(
            r#"(?i)(?:https?://|www\.)[^{SPACE}{CJK}]+?(?=[.,;:!?)\]\}}>'"”’»]*(?![^{SPACE}{CJK}]))|[a-zİıſK0-9._%+-]+@[a-zİıſK0-9.-]+\.[a-zİıſK]{{2,}}|@[a-zİıſK0-9_]+|[{CJK}]|{alnum}+(?:[-_]{alnum}+)*|[^{SPACE}]"#
        )).context("text regex")?;
        let body = format!(r#"[^{SPACE}\]\)\}}>「」『』"'“”{CJK}]"#);
        let unit = Regex::new(&format!(
            r"(?:https?://|www\.){body}+?(?=[.,;:!?]*(?!{body}))|(?:(?![{CJK}])[\p{{L}}\p{{N}}]|[@._\-+])+|[^{SPACE}]"
        )).context("text regex")?;
        Ok(Self {
            config,
            calibration,
            tokenizer,
            word,
            unit,
            cache: Mutex::new(HashMap::new()),
            batch_size: 8,
            token_budget: 16384,
        })
    }
    fn ids(&self, s: &str) -> Result<Vec<u32>> {
        // The cache holds no invariant a panic could break, so a poisoned
        // lock is still usable.
        if let Some(ids) = self.cache.lock().unwrap_or_else(|e| e.into_inner()).get(s) {
            return Ok(ids.clone());
        }
        let encoded = self
            .tokenizer
            .encode(s, false)
            .map_err(rt_core::Error::Tokenizer)?;
        let ids = encoded.get_ids().to_vec();
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() >= ID_CACHE_ENTRIES {
            cache.clear();
        }
        cache.insert(s.into(), ids.clone());
        Ok(ids)
    }
    fn part(&self, p: &Value, span: bool) -> Result<Part> {
        let s = text(&p["text"])?;
        let e = self
            .tokenizer
            .encode_char_offsets(s, false)
            .map_err(rt_core::Error::Tokenizer)?;
        let offsets = e.get_offsets();
        let mut words = None;
        if span {
            let boundary = chars_to_bytes(s);
            let mut byte_to_char = vec![0; s.len() + 1];
            for (c, &b) in boundary.iter().enumerate() {
                byte_to_char[b] = c;
            }
            let mut cum = Vec::with_capacity(offsets.len());
            let mut end = -1i64;
            for &(a, b) in offsets {
                if b > a {
                    end = end.max(b as i64);
                }
                cum.push(end);
            }
            let mut result = vec![];
            for m in self.word.find_iter(s) {
                let m = m.context("text regex")?;
                let (a, b) = (byte_to_char[m.start()], byte_to_char[m.end()]);
                let mut t = cum.partition_point(|&v| v <= a as i64);
                while t < offsets.len() && offsets[t].1 <= offsets[t].0 {
                    t += 1;
                }
                if t < offsets.len() && offsets[t].0 < b {
                    result.push(Word {
                        offset: [a, b],
                        first: t,
                    });
                }
            }
            words = Some(result);
        }
        Ok(Part {
            role: text(&p["role"])?.into(),
            ids: e.get_ids().to_vec(),
            words,
        })
    }
    fn assemble(&self, parts: &[Part], questions: &[Value]) -> Result<Record> {
        let mut built = vec![];
        let mut entities = vec![];
        let mut span_role = None;
        let mut protected = HashSet::new();
        for q in questions {
            let kind = text(&q["type"])?;
            let qt = self.ids(q["text"].as_str().unwrap_or(""))?;
            for role in roles(&q["over"])? {
                protected.insert(role);
            }
            if kind == "span" {
                check_shape!(span_role.is_none(), "one span question per row");
                span_role = Some(text(&q["over"])?.to_string());
                for (name, desc) in names(&q["labels"])? {
                    let ids = self.ids(&option_text(&name, &desc))?;
                    entities.push((name, ids));
                }
                continue;
            }
            check_shape!(
                ["choice", "score", "set"].contains(&kind),
                "unsupported question type {kind}"
            );
            let options = if kind == "score" {
                array(&q["levels"])?
                    .iter()
                    .map(|v| {
                        Ok((
                            text(v)?.to_string(),
                            q["options"][text(v)?].as_str().unwrap_or("").to_string(),
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                names(&q["options"])?
            };
            let mut opts = vec![];
            for (name, desc) in options {
                opts.push((name.clone(), self.ids(&option_text(&name, &desc))?));
            }
            let abstain = kind == "choice" && truth(&q["abstain"]);
            let length = 1
                + qt.len()
                + opts.iter().map(|(_, t)| 1 + t.len()).sum::<usize>()
                + usize::from(abstain);
            built.push(BuiltQuestion {
                qid: text(&q["id"])?.into(),
                qtype: kind.into(),
                over: q["over"].clone(),
                text: qt,
                options: opts,
                abstain,
                length,
            });
        }
        let mut segments = parts
            .iter()
            .map(|p| (p.role.clone(), p.ids.clone()))
            .collect::<Vec<_>>();
        let fixed = 3 + segments.len();
        let schema = built.iter().map(|q| q.length).sum::<usize>()
            + entities.iter().map(|(_, t)| 1 + t.len()).sum::<usize>();
        let max = self.config["max_length"].as_u64().context("max_length")? as usize;
        let over = |segments: &[(String, Vec<u32>)]| {
            (fixed + schema + segments.iter().map(|(_, t)| t.len()).sum::<usize>())
                .saturating_sub(max)
        };
        if let Some(i) = segments.iter().position(|(r, _)| !protected.contains(r)) {
            let n = segments[i].1.len();
            let keep = n.min(64).max(n.saturating_sub(over(&segments)));
            segments[i].1.truncate(keep);
            if over(&segments) > 0 {
                let keep = 1.max(segments[i].1.len().saturating_sub(over(&segments)));
                segments[i].1.truncate(keep);
            }
        }
        let mut protected_cut = 0;
        for i in 0..segments.len() {
            if protected.contains(&segments[i].0) && over(&segments) > 0 {
                let n = segments[i].1.len();
                let keep = 1.max(n.saturating_sub(over(&segments)));
                protected_cut += n as i64 - keep as i64;
                segments[i].1.truncate(keep);
            }
        }
        check_shape!(
            over(&segments) == 0,
            "schema alone does not fit max_length ({} tokens over)",
            over(&segments)
        );
        let marker = |name: &str| -> Result<u32> {
            Ok(self.config["marker_ids"][name]
                .as_u64()
                .with_context(|| format!("unknown marker {name}"))? as u32)
        };
        let mut ids = vec![self.config["bos_token_id"].as_u64().context("BOS")? as u32];
        let mut qs = vec![];
        for q in built {
            let mut rec = Question {
                qid: q.qid,
                qtype: q.qtype,
                over: q.over,
                position: ids.len(),
                options: vec![],
                names: vec![],
                abs: None,
                pool: [0, 0],
            };
            ids.push(marker("[Q]")?);
            ids.extend(q.text);
            for (name, toks) in q.options {
                rec.options.push(ids.len());
                rec.names.push(name);
                ids.push(marker("[O]")?);
                ids.extend(toks);
            }
            if q.abstain {
                rec.abs = Some(ids.len());
                ids.push(marker("[ABS]")?);
            }
            qs.push(rec);
        }
        let mut entity_positions = vec![];
        let mut entity_names = vec![];
        for (name, toks) in entities {
            entity_positions.push(ids.len());
            entity_names.push(name);
            ids.push(marker("[E]")?);
            ids.extend(toks);
        }
        ids.push(marker("[SEP_SCHEMA]")?);
        let mut ranges = HashMap::new();
        let mut lengths = HashMap::new();
        for (role, toks) in segments {
            let start = ids.len();
            let len = toks.len();
            ids.push(marker(&format!("[SEG_{role}]"))?);
            ids.extend(toks);
            ranges.insert(role.clone(), [start, ids.len()]);
            lengths.insert(role, len);
        }
        ids.push(self.config["eos_token_id"].as_u64().context("EOS")? as u32);
        for q in &mut qs {
            let r = roles(&q.over)?;
            check_shape!(!r.is_empty(), "empty target roles");
            q.pool = [
                ranges[&r[0]][0],
                ranges[r.last().context("target roles")?][1],
            ];
        }
        let mut word_positions = None;
        let mut word_offsets = vec![];
        if let Some(role) = span_role {
            let p = parts.iter().find(|p| p.role == role).context("span part")?;
            let mut positions = vec![];
            for word in p.words.as_ref().context("span words")? {
                if word.first < lengths[&role] {
                    word_offsets.push(word.offset);
                    positions.push(ranges[&role][0] + 1 + word.first);
                }
            }
            word_positions = Some(positions);
        }
        Ok(Record {
            ids,
            questions: qs,
            names: entity_names,
            entities: entity_positions,
            word_positions,
            word_offsets,
            segment_lengths: lengths,
            protected_cut,
        })
    }
    pub fn prepare(&self, rows: &[Value]) -> Result<Plan> {
        check_shape!(!rows.is_empty(), "empty prepare batch");
        check_shape!(
            self.batch_size > 0 && self.token_budget > 0,
            "invalid batching"
        );
        let mut jobs = vec![];
        let mut all_tokens = vec![];
        let mut full_words = vec![];
        for (row_index, row) in rows.iter().enumerate() {
            let questions = array(&row["questions"])?;
            let span = questions.iter().find(|q| q["type"] == "span");
            let span_role = span.map(|q| text(&q["over"])).transpose()?;
            let mut parts = vec![];
            let mut unique = HashSet::new();
            for p in array(&row["parts"])? {
                let role = text(&p["role"])?;
                check_shape!(unique.insert(role), "one part per role");
                parts.push(self.part(p, span_role == Some(role))?);
            }
            for q in questions {
                for role in roles(&q["over"])? {
                    check_shape!(
                        parts.iter().any(|p| p.role == role),
                        "question target role {role} missing"
                    );
                }
            }
            all_tokens.push(
                parts
                    .iter()
                    .map(|p| (p.role.clone(), p.ids.len()))
                    .collect(),
            );
            full_words.push(
                parts
                    .iter()
                    .find_map(|p| {
                        p.words
                            .as_ref()
                            .map(|w| w.iter().map(|w| w.offset).collect())
                    })
                    .unwrap_or_default(),
            );
            let rec = self.assemble(&parts, questions)?;
            let common = questions
                .first()
                .and_then(|q| q["over"].as_str())
                .filter(|r| questions.iter().all(|q| q["over"] == *r));
            let role = span_role.or(common);
            if rec.protected_cut != 0 && role.is_some() {
                let role = role.context("window role")?;
                let part_index = parts
                    .iter()
                    .position(|p| p.role == role)
                    .context("window role")?;
                let part = &parts[part_index];
                let n = part.ids.len();
                let len = rec.segment_lengths[role];
                if len < n {
                    let overlap = (self.config["window_overlap"].as_u64().context("overlap")?
                        as usize)
                        .min(len / 4);
                    let stride = len.saturating_sub(overlap).max(1);
                    for start in (0..n.saturating_sub(overlap).max(1)).step_by(stride) {
                        let end = n.min(start + len);
                        let mut sliced = parts.clone();
                        let mut segment = part.clone();
                        segment.ids = part.ids[start..end].to_vec();
                        let mut global = None;
                        if let Some(words) = &part.words {
                            let selected = words
                                .iter()
                                .enumerate()
                                .filter(|(_, w)| w.first >= start && w.first < end)
                                .collect::<Vec<_>>();
                            global = Some(selected.iter().map(|(i, _)| *i).collect());
                            segment.words = Some(
                                selected
                                    .iter()
                                    .map(|(_, w)| Word {
                                        offset: w.offset,
                                        first: w.first - start,
                                    })
                                    .collect(),
                            );
                        }
                        sliced[part_index] = segment;
                        jobs.push(Job {
                            row: row_index,
                            windowed: true,
                            global_words: global,
                            record: self.assemble(&sliced, questions)?,
                        });
                        if end >= n {
                            break;
                        }
                    }
                    continue;
                }
            }
            jobs.push(Job {
                row: row_index,
                windowed: false,
                global_words: None,
                record: rec,
            });
        }
        let mut order = (0..jobs.len()).collect::<Vec<_>>();
        order.sort_by_key(|&i| jobs[i].record.ids.len());
        let mut batches = vec![];
        let mut current = vec![];
        let mut width = 0;
        for j in order {
            let w = width.max(jobs[j].record.ids.len());
            if !current.is_empty()
                && (current.len() + 1 > self.batch_size
                    || w * (current.len() + 1) > self.token_budget)
            {
                batches.push(self.batch(&jobs, &current)?);
                current.clear();
                width = 0;
            }
            width = width.max(jobs[j].record.ids.len());
            current.push(j);
        }
        if !current.is_empty() {
            batches.push(self.batch(&jobs, &current)?);
        }
        Ok(Plan {
            batches,
            jobs,
            rows: rows.to_vec(),
            tokens: all_tokens,
            full_words,
        })
    }
    fn batch(&self, jobs: &[Job], indices: &[usize]) -> Result<Batch> {
        let width = indices
            .iter()
            .map(|&i| jobs[i].record.ids.len())
            .max()
            .context("empty batch")?;
        let pad = self.config["pad_token_id"].as_u64().context("pad")? as u32;
        let mut input = Inputs {
            input_ids: vec![vec![pad; width]; indices.len()],
            attention_mask: vec![vec![0; width]; indices.len()],
            q_index: vec![],
            opt_index: vec![],
            unit_index: vec![],
            ent_index: vec![],
        };
        for (b, &i) in indices.iter().enumerate() {
            let r = &jobs[i].record;
            input.input_ids[b][..r.ids.len()].copy_from_slice(&r.ids);
            input.attention_mask[b][..r.ids.len()].fill(1);
            for q in &r.questions {
                let owner = input.q_index.len();
                input.q_index.push([b, q.position, q.pool[0], q.pool[1]]);
                input
                    .opt_index
                    .extend(q.options.iter().chain(q.abs.iter()).map(|&p| [b, p, owner]));
            }
            if let Some(positions) = &r.word_positions {
                input.unit_index.extend(positions.iter().map(|&p| [b, p]));
                input.ent_index.extend(r.entities.iter().map(|&p| [b, p]));
            }
        }
        if input.q_index.is_empty() {
            input.q_index.push([0, 0, 0, 1]);
        }
        if input.opt_index.is_empty() {
            input.opt_index.push([0, 0, 0]);
        }
        if input.unit_index.is_empty() {
            input.unit_index.push([0, 0]);
        }
        if input.ent_index.is_empty() {
            input.ent_index.push([0, 0]);
        }
        Ok(Batch {
            jobs: indices.to_vec(),
            inputs: input,
        })
    }
}

// Validate externally deserialized plans before indexing any reconstruction
// arrays. Prepared plans have exactly one batch assignment per job and consistent
// per-row window schemas; malformed plans return Err before reconstruction.
fn validate_plan(plan: &Plan, outputs: &[Logits]) -> Result<()> {
    check_shape!(outputs.len() == plan.batches.len(), "batch output count");
    check_shape!(
        plan.tokens.len() == plan.rows.len() && plan.full_words.len() == plan.rows.len(),
        "plan row metadata shape"
    );
    let mut seen = vec![false; plan.jobs.len()];
    for (batch, out) in plan.batches.iter().zip(outputs) {
        check_shape!(!batch.jobs.is_empty(), "empty batch jobs");
        check_shape!(
            batch.inputs.input_ids.len() == batch.jobs.len()
                && batch.inputs.attention_mask.len() == batch.jobs.len(),
            "batch input rows"
        );
        check_shape!(
            out.opt_logits.len() == batch.inputs.opt_index.len(),
            "option output shape"
        );
        check_shape!(
            out.span_logits.len() == batch.inputs.unit_index.len()
                && out
                    .span_logits
                    .iter()
                    .all(|row| row.len() == batch.inputs.ent_index.len()),
            "span output shape"
        );
        let mut options = 0usize;
        let mut units = 0usize;
        let mut entities = 0usize;
        for &index in &batch.jobs {
            let job = plan
                .jobs
                .get(index)
                .context("batch job index out of bounds")?;
            check_shape!(!seen[index], "duplicate batch job");
            seen[index] = true;
            check_shape!(job.row < plan.rows.len(), "job row index out of bounds");
            for q in &job.record.questions {
                options = options
                    .checked_add(q.options.len())
                    .and_then(|n| n.checked_add(usize::from(q.abs.is_some())))
                    .context("option count overflow")?;
            }
            if let Some(positions) = &job.record.word_positions {
                check_shape!(
                    positions.len() == job.record.word_offsets.len()
                        && job.record.entities.len() == job.record.names.len(),
                    "word/entity metadata shape"
                );
                units = units
                    .checked_add(positions.len())
                    .context("unit count overflow")?;
                entities = entities
                    .checked_add(job.record.entities.len())
                    .context("entity count overflow")?;
                if job.windowed {
                    let global = job.global_words.as_ref().context("window word indices")?;
                    check_shape!(
                        global.len() == positions.len()
                            && global.iter().all(|&g| g < plan.full_words[job.row].len()),
                        "window mapping out of bounds"
                    );
                }
            }
        }
        check_shape!(
            options.max(1) == batch.inputs.opt_index.len()
                && units.max(1) == batch.inputs.unit_index.len()
                && entities.max(1) == batch.inputs.ent_index.len(),
            "plan head metadata shape"
        );
    }
    check_shape!(seen.iter().all(|&v| v), "unassigned plan job");
    for row in 0..plan.rows.len() {
        let jobs = plan
            .jobs
            .iter()
            .filter(|j| j.row == row)
            .collect::<Vec<_>>();
        let first = jobs.first().context("row has no jobs")?;
        for job in &jobs {
            check_shape!(
                job.windowed == first.windowed
                    && job.record.names == first.record.names
                    && job.record.word_positions.is_some() == first.record.word_positions.is_some(),
                "inconsistent window metadata"
            );
            check_shape!(
                job.record.questions.len() == first.record.questions.len(),
                "inconsistent question count"
            );
            for (q, r) in job.record.questions.iter().zip(&first.record.questions) {
                check_shape!(
                    q.qid == r.qid
                        && q.qtype == r.qtype
                        && q.names == r.names
                        && q.options.len() == r.options.len()
                        && q.abs.is_some() == r.abs.is_some(),
                    "inconsistent window question schema"
                );
            }
        }
    }
    Ok(())
}

#[derive(Clone)]
struct JobOutput {
    questions: Vec<Vec<f32>>,
    span: Option<Vec<Vec<f32>>>,
}
impl Frontend {
    pub fn finish(&self, plan: &Plan, outputs: &[Logits]) -> Result<Vec<Value>> {
        validate_plan(plan, outputs)?;
        let mut results = vec![None; plan.jobs.len()];
        for (batch, out) in plan.batches.iter().zip(outputs) {
            let mut option = 0;
            let mut unit = 0;
            let mut entity = 0;
            for &j in &batch.jobs {
                let r = &plan.jobs[j].record;
                let mut questions = vec![];
                for q in &r.questions {
                    let n = q.options.len() + usize::from(q.abs.is_some());
                    check_shape!(option + n <= out.opt_logits.len(), "option output shape");
                    questions.push(out.opt_logits[option..option + n].to_vec());
                    option += n;
                }
                let span = if let Some(pos) = &r.word_positions {
                    let mut rows = vec![];
                    check_shape!(
                        unit + pos.len() <= out.span_logits.len(),
                        "span output rows"
                    );
                    for row in &out.span_logits[unit..unit + pos.len()] {
                        check_shape!(
                            entity + r.entities.len() <= row.len(),
                            "span output columns"
                        );
                        rows.push(row[entity..entity + r.entities.len()].to_vec());
                    }
                    unit += pos.len();
                    entity += r.entities.len();
                    Some(rows)
                } else {
                    None
                };
                results[j] = Some(JobOutput { questions, span });
            }
        }
        let mut final_rows = vec![];
        for row in 0..plan.rows.len() {
            let indices = plan
                .jobs
                .iter()
                .enumerate()
                .filter(|(_, j)| j.row == row)
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            let first = *indices.first().context("row has no jobs")?;
            let rec = &plan.jobs[first].record;
            let mut questions = vec![];
            for (qi, q) in rec.questions.iter().enumerate() {
                let n = q.options.len() + usize::from(q.abs.is_some());
                let mut logits = vec![
                    if q.qtype == "set" {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    };
                    n
                ];
                for &j in &indices {
                    let v = results[j]
                        .as_ref()
                        .context("missing output")?
                        .questions
                        .get(qi)
                        .context("missing question output")?;
                    check_shape!(v.len() == n, "question output size");
                    for k in 0..n {
                        if q.qtype == "set" {
                            logits[k] = logits[k].max(v[k]);
                        } else {
                            logits[k] += v[k];
                        }
                    }
                }
                if q.qtype != "set" {
                    for x in &mut logits {
                        *x /= indices.len() as f32;
                    }
                }
                questions.push(json!({"qid":q.qid,"qtype":q.qtype,"names":q.names,"logits":logits.into_iter().map(f64::from).collect::<Vec<_>>(),"has_abs":q.abs.is_some()}));
            }
            let span = if rec.word_positions.is_some() {
                let (offs, logits) = if indices.len() == 1 && !plan.jobs[first].windowed {
                    (
                        rec.word_offsets.clone(),
                        results[first]
                            .as_ref()
                            .context("missing output")?
                            .span
                            .as_ref()
                            .context("span output")?
                            .iter()
                            .map(|r| r.iter().copied().map(f64::from).collect::<Vec<_>>())
                            .collect::<Vec<_>>(),
                    )
                } else {
                    let offs = plan.full_words[row].clone();
                    let mut accumulated = vec![vec![0f64; rec.names.len()]; offs.len()];
                    let mut counts = vec![0usize; offs.len()];
                    for &j in &indices {
                        let global = plan.jobs[j]
                            .global_words
                            .as_ref()
                            .context("window word indices")?;
                        let values = results[j]
                            .as_ref()
                            .context("missing output")?
                            .span
                            .as_ref()
                            .context("window span outputs")?;
                        check_shape!(values.len() <= global.len(), "window mapping");
                        for (&g, v) in global.iter().zip(values) {
                            let count =
                                counts.get_mut(g).context("window mapping out of bounds")?;
                            *count += 1;
                            check_shape!(v.len() == rec.names.len(), "window entity output shape");
                            for k in 0..rec.names.len() {
                                accumulated[g][k] += v[k] as f64;
                            }
                        }
                    }
                    for i in 0..offs.len() {
                        for x in &mut accumulated[i] {
                            *x = if counts[i] == 0 {
                                -30.
                            } else {
                                *x / counts[i] as f64
                            };
                        }
                    }
                    (offs, accumulated)
                };
                json!({"names":rec.names,"offs":offs,"logits":logits})
            } else {
                Value::Null
            };
            final_rows.push(json!({"qs":questions,"span":span,"n_tokens":plan.tokens[row],"windows":if plan.jobs[first].windowed{indices.len()}else{0},"n_input":indices.iter().map(|&j|plan.jobs[j].record.ids.len()).sum::<usize>()}));
        }
        Ok(final_rows)
    }
    pub fn predict_raw<F>(&self, rows: &[Value], mut execute: F) -> Result<Vec<Value>>
    where
        F: FnMut(&Inputs) -> Result<Logits>,
    {
        let plan = self.prepare(rows)?;
        let mut outputs = vec![];
        for batch in &plan.batches {
            outputs.push(execute(&batch.inputs)?);
        }
        self.finish(&plan, &outputs)
    }
    pub fn temperature(&self, kind: &str) -> f64 {
        self.calibration["temperature"][kind].as_f64().unwrap_or(1.)
    }
    fn span_threshold(&self, id: &str, names: &[String], tokens: usize) -> Result<(f64, String)> {
        let thresholds = &self.calibration["thresholds"];
        let pii = id == "pii"
            || (!names.is_empty()
                && names.iter().all(|n| {
                    self.calibration["pii_types"]
                        .as_array()
                        .map(|a| a.iter().any(|v| v == n))
                        .unwrap_or(false)
                }));
        if pii {
            if let Some(anchors) = self.calibration["pii_length_rule"]["anchors"].as_array() {
                check_shape!(!anchors.is_empty(), "PII anchors");
                let x = (tokens.max(1) as f64).ln();
                let mut y = anchors[0]["threshold"].as_f64().context("threshold")?.ln();
                for pair in anchors.windows(2) {
                    let lo = pair[0]["n_tokens"].as_f64().context("anchor")?.ln();
                    let hi = pair[1]["n_tokens"].as_f64().context("anchor")?.ln();
                    let yl = pair[0]["threshold"].as_f64().context("PII threshold")?.ln();
                    let yh = pair[1]["threshold"].as_f64().context("PII threshold")?.ln();
                    if x >= lo {
                        y = if x >= hi {
                            yh
                        } else {
                            yl + (yh - yl) * (x - lo) / (hi - lo)
                        };
                    }
                }
                return Ok((y.exp(), "pii:length".into()));
            }
            if let Some(buckets) = self.calibration["pii_buckets"].as_array() {
                for b in buckets {
                    if tokens as f64 <= b["max_tokens"].as_f64().unwrap_or(f64::INFINITY) {
                        return Ok((
                            b["threshold"].as_f64().context("bucket")?,
                            format!("pii:{}", text(&b["name"])?),
                        ));
                    }
                }
            }
        }
        for family in ["halu", "toxic"] {
            if id == family {
                return Ok((
                    thresholds[format!("span:{family}")]
                        .as_f64()
                        .context("family threshold")?,
                    family.into(),
                ));
            }
        }
        let key = format!("span:{id}");
        let specific = thresholds.get(&key);
        Ok((
            specific
                .unwrap_or(&thresholds["span:*"])
                .as_f64()
                .context("span threshold")?,
            if specific.is_some() {
                key
            } else {
                "span:*".into()
            },
        ))
    }
    pub fn calibrated(&self, raw: &Value, row: &Value, thresholds: &Value) -> Result<Vec<Value>> {
        let schemas = array(&row["questions"])?;
        let mut answers = vec![];
        for q in array(&raw["qs"])? {
            let id = text(&q["qid"])?;
            let kind = text(&q["qtype"])?;
            let names = string_array(&q["names"])?;
            check_shape!(!names.is_empty(), "empty decision options");
            let logits = number_array(&q["logits"])?;
            let n = names.len();
            check_shape!(logits.len() >= n, "decision logits");
            let temp = self.temperature(kind);
            let probabilities = if kind == "set" {
                logits[..n]
                    .iter()
                    .map(|&x| sigmoid(x / temp))
                    .collect::<Vec<_>>()
            } else {
                softmax(&logits[..n].iter().map(|&x| x / temp).collect::<Vec<_>>())
            };
            let probs: serde_json::Map<String, Value> = names
                .iter()
                .cloned()
                .zip(probabilities.iter().copied().map(Value::from))
                .collect();
            if kind == "set" {
                let key = format!("set:{id}");
                let threshold = thresholds[id]
                    .as_f64()
                    .or_else(|| self.calibration["thresholds"][key].as_f64())
                    .or_else(|| self.calibration["thresholds"]["set:*"].as_f64())
                    .context("set threshold")?;
                answers.push(json!({"id":id,"type":kind,"answer":names.iter().zip(&probabilities).filter(|(_,p)|**p>threshold).map(|(n,_)|n).collect::<Vec<_>>(),"probabilities":probs,"threshold":threshold}));
                continue;
            }
            let winner = argmax(&logits[..n]);
            let mut answer =
                json!({"id":id,"type":kind,"answer":names[winner],"probabilities":probs});
            if q["has_abs"] == true {
                answer["abstain_probability"] = json!(*softmax(
                    &logits.iter().map(|&x| x / temp).collect::<Vec<_>>()
                )
                .last()
                .context("abstain")?);
            }
            if kind == "score" {
                let schema = schemas
                    .iter()
                    .rev()
                    .find(|s| s["id"] == id)
                    .context("score schema")?;
                let values = &schema["values"];
                answer["expected_index"] = json!(probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * p)
                    .sum::<f64>());
                answer["score"] = json!(names
                    .iter()
                    .zip(&probabilities)
                    .enumerate()
                    .map(|(i, (name, p))| values[name].as_f64().unwrap_or(if n > 1 {
                        i as f64 / (n - 1) as f64
                    } else {
                        0.
                    }) * p)
                    .sum::<f64>());
            }
            answers.push(answer);
        }
        if !raw["span"].is_null() {
            let schema = schemas
                .iter()
                .find(|s| s["type"] == "span")
                .context("span schema")?;
            let id = text(&schema["id"])?;
            let role = text(&schema["over"])?;
            let text_value = row["parts"]
                .as_array()
                .context("parts")?
                .iter()
                .find(|p| p["role"] == role)
                .context("span part")?;
            let source = text(&text_value["text"])?;
            let span = &raw["span"];
            let labels = string_array(&span["names"])?;
            let offsets: Vec<[usize; 2]> = serde_json::from_value(span["offs"].clone())?;
            let logits: Vec<Vec<f64>> = serde_json::from_value(span["logits"].clone())?;
            let probabilities = logits
                .iter()
                .map(|r| {
                    r.iter()
                        .map(|&x| sigmoid(x / self.temperature("span")))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let (mut threshold, mut rule) = if let Some(t) = thresholds[id].as_f64() {
                (t, "given".into())
            } else {
                self.span_threshold(
                    id,
                    &labels,
                    raw["n_tokens"][role].as_u64().context("token count")? as usize,
                )?
            };
            if rule.starts_with("pii:") {
                let gate = &self.calibration["pii_sparse_gate"];
                if let (Some(k), Some(sparse)) = (gate["K"].as_i64(), gate["t_sparse"].as_f64()) {
                    if k >= 0
                        && threshold < sparse
                        && !offsets.is_empty()
                        && self
                            .decode_spans(
                                &probabilities,
                                &offsets,
                                &labels,
                                source,
                                gate["probe_threshold"].as_f64().unwrap_or(0.5),
                            )?
                            .len()
                            <= k as usize
                    {
                        threshold = sparse;
                        rule += "+sparse";
                    }
                }
            }
            let spans = self.decode_spans(&probabilities, &offsets, &labels, source, threshold)?;
            answers.push(json!({"id":id,"type":"span","target_part":role,"spans":spans,"threshold":threshold,"threshold_rule":rule}));
        }
        Ok(answers)
    }
    pub fn decode_spans(
        &self,
        probs: &[Vec<f64>],
        offsets: &[[usize; 2]],
        labels: &[String],
        source: &str,
        threshold: f64,
    ) -> Result<Vec<Value>> {
        if offsets.is_empty() {
            return Ok(vec![]);
        }
        check_shape!(
            probs.len() == offsets.len()
                && !labels.is_empty()
                && probs.iter().all(|r| r.len() == labels.len()),
            "span probability shape"
        );
        let chars = source.chars().collect::<Vec<_>>();
        let bytes = chars_to_bytes(source);
        let mut char_at_byte = vec![0; source.len() + 1];
        for (i, &b) in bytes.iter().enumerate() {
            char_at_byte[b] = i;
        }
        let mut units = vec![-1i64; chars.len() + 1];
        for (k, m) in self.unit.find_iter(source).enumerate() {
            let m = m.context("text regex")?;
            for u in &mut units[char_at_byte[m.start()]..char_at_byte[m.end()]] {
                *u = k as i64;
            }
        }
        let p = probs
            .iter()
            .map(|r| r.iter().map(|&v| v as f32).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut lab = vec![-1i64; offsets.len()];
        let mut peaks = vec![0f32; offsets.len()];
        let mut token_units = vec![-1i64; offsets.len()];
        let valid = offsets.iter().map(|o| o[1] > o[0]).collect::<Vec<_>>();
        for i in 0..offsets.len() {
            let [a, b] = offsets[i];
            check_shape!(
                a <= b && b <= chars.len(),
                "span offset [{a}, {b}] beyond text length {}",
                chars.len()
            );
            if !valid[i] {
                continue;
            }
            let k = argmax_f32(&p[i]);
            peaks[i] = p[i][k];
            if peaks[i] > threshold as f32 {
                lab[i] = k as i64;
            }
            let mut f = a;
            while f < b && python_space(chars[f]) {
                f += 1;
            }
            if f < b {
                token_units[i] = units[f];
            }
        }
        let mut t = 0;
        while t < offsets.len() {
            if token_units[t] < 0 {
                t += 1;
                continue;
            }
            let mut end = t;
            while end + 1 < offsets.len()
                && (token_units[end + 1] == token_units[t] || !valid[end + 1])
            {
                end += 1;
            }
            let selected = (t..=end)
                .filter(|&i| valid[i] && lab[i] >= 0)
                .collect::<Vec<_>>();
            if !selected.is_empty() {
                let mut order = vec![];
                let mut scores = vec![0f64; labels.len()];
                for &i in &selected {
                    let c = lab[i] as usize;
                    if !order.contains(&c) {
                        order.push(c);
                    }
                    scores[c] += peaks[i] as f64;
                }
                if order.len() > 1 {
                    let mut winner = order[0];
                    for &c in &order[1..] {
                        if scores[c] > scores[winner] {
                            winner = c;
                        }
                    }
                    for &i in &selected {
                        lab[i] = winner as i64;
                    }
                }
                let present = selected.iter().map(|&i| lab[i]).collect::<HashSet<_>>();
                for c in present {
                    let positions = selected
                        .iter()
                        .copied()
                        .filter(|&i| lab[i] == c)
                        .collect::<Vec<_>>();
                    let lo = positions[0];
                    let hi = *positions.last().context("span positions")?;
                    for i in lo + 1..hi {
                        if valid[i] && lab[i] < 0 {
                            lab[i] = c;
                        }
                    }
                }
            }
            t = end + 1;
        }
        let mut groups: Vec<(usize, usize, usize, Vec<usize>)> = vec![];
        let mut current: Option<(usize, usize, usize, Vec<usize>)> = None;
        for i in 0..offsets.len() {
            if !valid[i] {
                continue;
            }
            if let Some(g) = &mut current {
                if lab[i] == g.0 as i64 {
                    g.2 = offsets[i][1];
                    g.3.push(i);
                    continue;
                }
            }
            if let Some(g) = current.take() {
                groups.push(g);
            }
            if lab[i] >= 0 {
                current = Some((lab[i] as usize, offsets[i][0], offsets[i][1], vec![i]));
            }
        }
        if let Some(g) = current {
            groups.push(g);
        }
        let mut result = vec![];
        for (c, start, end, indices) in groups {
            let (start, end) = trim(source, start, end)?;
            if end <= start {
                continue;
            }
            let sum = indices.iter().map(|&i| p[i][c]).sum::<f32>();
            let probability = sum / indices.len() as f32;
            result.push(json!({"start":start,"end":end,"label":labels[c],"probability":probability as f64,"text":&source[bytes[start]..bytes[end]]}));
        }
        Ok(result)
    }
}
fn python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
fn is_cjk(c: char) -> bool {
    matches!(c as u32,0x2e80..=0x2fff|0x3000..=0x303f|0x3040..=0x30ff|0x3100..=0x31ff|0x3400..=0x4dbf|0x4e00..=0x9fff|0xa000..=0xa4cf|0xac00..=0xd7af|0xf900..=0xfaff|0xfe30..=0xfe4f|0xff00..=0xffef)
}
pub fn trim(source: &str, mut start: usize, mut end: usize) -> Result<(usize, usize)> {
    let count = source.chars().count();
    check_shape!(
        start <= count && end <= count,
        "trim offsets [{start}, {end}] exceed text length {count}"
    );
    if end <= start {
        return Ok((start, end));
    }
    let chars = source.chars().collect::<Vec<_>>();
    while start < end && (python_space(chars[start]) || EDGE.contains(chars[start])) {
        start += 1;
    }
    while end > start && (python_space(chars[end - 1]) || EDGE.contains(chars[end - 1])) {
        end -= 1;
    }
    let first = chars[start..end].iter().collect::<String>().to_lowercase();
    if first.starts_with("http://") || first.starts_with("https://") || first.starts_with("www.") {
        for (i, &ch) in chars.iter().enumerate().take(end).skip(start) {
            if is_cjk(ch) {
                end = i;
                break;
            }
        }
        while end > start
            && (python_space(chars[end - 1])
                || EDGE.contains(chars[end - 1])
                || ".,;:!?)".contains(chars[end - 1]))
        {
            end -= 1;
        }
    }
    Ok((start, end))
}
fn string_array(value: &Value) -> Result<Vec<String>> {
    array(value)?.iter().map(|v| Ok(text(v)?.into())).collect()
}
fn number_array(value: &Value) -> Result<Vec<f64>> {
    array(value)?
        .iter()
        .map(|v| v.as_f64().context("expected number"))
        .collect()
}
fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
fn argmax(x: &[f64]) -> usize {
    let mut k = 0;
    for i in 1..x.len() {
        if x[i] > x[k] {
            k = i;
        }
    }
    k
}
fn argmax_f32(x: &[f32]) -> usize {
    let mut k = 0;
    for i in 1..x.len() {
        if x[i] > x[k] {
            k = i;
        }
    }
    k
}
fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut p = x.iter().map(|v| (*v - max).exp()).collect::<Vec<_>>();
    let sum = p.iter().sum::<f64>();
    for v in &mut p {
        *v /= sum;
    }
    p
}

fn named_options(value: &Value) -> Result<Value> {
    if value.is_object() {
        return Ok(value.clone());
    }
    let mut out = serde_json::Map::new();
    for v in array(value)? {
        if let Some(name) = v.as_str() {
            out.insert(name.into(), json!(""));
        } else {
            out.insert(
                text(&v["name"])?.into(),
                json!(v["description"].as_str().unwrap_or("")),
            );
        }
    }
    Ok(Value::Object(out))
}
fn canonical(value: &Value) -> Result<String> {
    Ok(match value {
        Value::String(s) => serde_json::to_string(s)?,
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            // Integer tokens retain their exact decimal digits, including values
            // outside i64/u64. Only tokens containing fraction/exponent syntax
            // follow the released Python float64 canonicalization path.
            let token = n.to_string();
            if token.contains('.') || token.contains('e') || token.contains('E') {
                let x = n.as_f64().context("JSON number")?;
                check_shape!(x.is_finite(), "nonfinite JSON number");
                let a = x.abs();
                if a >= 1e16 || (a > 0. && a < 1e-4) {
                    let s = format!("{x:e}");
                    let (mantissa, exp) = s.split_once('e').context("scientific format")?;
                    let exponent: i32 = exp.parse().context("JSON exponent")?;
                    format!(
                        "{mantissa}e{}{:02}",
                        if exponent >= 0 { '+' } else { '-' },
                        exponent.abs()
                    )
                } else {
                    let s = x.to_string();
                    if s.contains('.') {
                        s
                    } else {
                        format!("{s}.0")
                    }
                }
            } else {
                token
            }
        }
        Value::Array(a) => format!(
            "[{}]",
            a.iter()
                .map(canonical)
                .collect::<Result<Vec<_>>>()?
                .join(",")
        ),
        Value::Object(o) => {
            let mut keys = o.keys().collect::<Vec<_>>();
            keys.sort();
            format!(
                "{{{}}}",
                keys.into_iter()
                    .map(|k| Ok(format!(
                        "{}:{}",
                        serde_json::to_string(k)?,
                        canonical(&o[k])?
                    )))
                    .collect::<Result<Vec<_>>>()?
                    .join(",")
            )
        }
    })
}
fn content(value: &Value) -> Result<String> {
    if let Some(s) = value.as_str() {
        Ok(s.into())
    } else {
        canonical(value)
    }
}
struct Prediction {
    response: Value,
    span_max: HashMap<String, f64>,
    input_tokens: usize,
}
impl Frontend {
    fn to_generic(&self, request: &Value) -> Result<(Vec<Value>, Vec<Value>, Value)> {
        let mut parts = vec![];
        for part in array(&request["parts"])? {
            let role = text(&part["type"])?;
            check_shape!(
                !self.config["marker_ids"][format!("[SEG_{role}]")].is_null(),
                "part type must be one of user / context / answer, got {role}"
            );
            parts.push(json!({"role":role,"text":part["text"]}));
        }
        check_shape!(!parts.is_empty(), "empty parts");
        let roles = parts
            .iter()
            .map(|p| text(&p["role"]))
            .collect::<Result<Vec<_>>>()?;
        let mut questions = vec![];
        let mut thresholds = serde_json::Map::new();
        for (i, q) in array(&request["questions"])?.iter().enumerate() {
            let kind = text(&q["type"])?;
            let id = q
                .get("id")
                .map(|v| {
                    if let Some(s) = v.as_str() {
                        s.into()
                    } else {
                        v.to_string()
                    }
                })
                .unwrap_or_else(|| format!("q{i}"));
            let target = if truth(&q["target_part"]) {
                q["target_part"].clone()
            } else {
                json!(if roles.contains(&"user") {
                    "user"
                } else {
                    roles[0]
                })
            };
            let mut generic =
                json!({"id":id,"type":kind,"over":target,"text":q["text"].as_str().unwrap_or("")});
            if kind == "span" {
                generic["labels"] = named_options(if truth(&q["labels"]) {
                    &q["labels"]
                } else {
                    &q["options"]
                })?;
            } else {
                let value = if truth(&q["options"]) {
                    &q["options"]
                } else if truth(&q["labels"]) {
                    &q["labels"]
                } else {
                    &q["levels"]
                };
                let opts = named_options(value)?;
                if kind == "score" {
                    generic["levels"] = json!(opts
                        .as_object()
                        .context("options object")?
                        .keys()
                        .collect::<Vec<_>>());
                    if truth(&q["values"]) {
                        generic["values"] = if let Some(vals) = q["values"].as_array() {
                            Value::Object(
                                opts.as_object()
                                    .context("options object")?
                                    .keys()
                                    .cloned()
                                    .zip(vals.iter().cloned())
                                    .collect(),
                            )
                        } else {
                            q["values"].clone()
                        };
                    }
                }
                if kind == "choice" {
                    generic["abstain"] = json!(q.get("abstain").map(truth).unwrap_or(true));
                }
                generic["options"] = opts;
            }
            if !q["threshold"].is_null() {
                thresholds.insert(id, q["threshold"].clone());
            }
            questions.push(generic);
        }
        Ok((parts, questions, Value::Object(thresholds)))
    }
    fn predict_inner<F>(
        &self,
        request: &Value,
        limits: &HashMap<String, [usize; 2]>,
        execute: &mut F,
    ) -> Result<Prediction>
    where
        F: FnMut(&Inputs) -> Result<Logits>,
    {
        let (parts, questions, thresholds) = self.to_generic(request)?;
        let spans = questions
            .iter()
            .filter(|q| q["type"] == "span")
            .cloned()
            .collect::<Vec<_>>();
        let mut primary = questions
            .iter()
            .filter(|q| q["type"] != "span")
            .cloned()
            .collect::<Vec<_>>();
        primary.extend(spans.iter().take(1).cloned());
        let mut rows = vec![];
        if !primary.is_empty() {
            rows.push(json!({"parts":parts,"questions":primary}));
        }
        for span in spans.iter().skip(1) {
            rows.push(json!({"parts":parts,"questions":[span]}));
        }
        check_shape!(!rows.is_empty(), "empty questions");
        let raws = self.predict_raw(&rows, execute)?;
        let mut by = HashMap::new();
        let mut span_max = HashMap::new();
        let mut windows = 0;
        let mut input_tokens = 0;
        for (raw, row) in raws.iter().zip(&rows) {
            windows = windows.max(raw["windows"].as_u64().context("windows")?);
            input_tokens += raw["n_input"].as_u64().context("input tokens")? as usize;
            for answer in self.calibrated(raw, row, &thresholds)? {
                by.insert(text(&answer["id"])?.to_string(), answer);
            }
            if !raw["span"].is_null() {
                let schema = array(&row["questions"])?
                    .iter()
                    .find(|q| q["type"] == "span")
                    .context("span schema")?;
                let id = text(&schema["id"])?;
                let span = &raw["span"];
                let offsets: Vec<[usize; 2]> = serde_json::from_value(span["offs"].clone())?;
                let logits: Vec<Vec<f64>> = serde_json::from_value(span["logits"].clone())?;
                let mut max = 0f64;
                for (offset, values) in offsets.iter().zip(logits) {
                    if let Some(&[lo, hi]) = limits.get(id) {
                        if offset[0] < lo || offset[1] > hi {
                            continue;
                        }
                    }
                    for value in values {
                        max = max.max(sigmoid(value / self.temperature("span")));
                    }
                }
                span_max.insert(id.into(), max);
            }
        }
        let answers = questions
            .iter()
            .map(|q| by.get(text(&q["id"])?).cloned().context("missing answer"))
            .collect::<Result<Vec<_>>>()?;
        Ok(Prediction {
            response: json!({"answers":answers,"meta":{"tokens":raws[0]["n_tokens"],"windows":windows,"sequences":rows.len()}}),
            span_max,
            input_tokens,
        })
    }
    pub fn predict<F>(&self, request: &Value, mut execute: F) -> Result<Value>
    where
        F: FnMut(&Inputs) -> Result<Logits>,
    {
        if let Some(requests) = request.as_array() {
            let mut out = vec![];
            for r in requests {
                out.push(
                    self.predict_inner(r, &HashMap::new(), &mut execute)?
                        .response,
                );
            }
            return Ok(json!(out));
        }
        Ok(self
            .predict_inner(request, &HashMap::new(), &mut execute)?
            .response)
    }
}

fn invalid<T>(location: Vec<Value>, message: impl Into<String>, kind: &str) -> Result<T> {
    Err(SystemOneError {
        location,
        message: message.into(),
        kind: kind.into(),
    }
    .into())
}
fn nonblank(value: &Value) -> bool {
    value
        .as_str()
        .map(|s| !s.chars().all(python_space))
        .unwrap_or(true)
}
fn state_role(key: &str) -> Option<&'static str> {
    match key {
        "request" | "user" | "prompt" => Some("user"),
        "source" | "context" | "document" | "tools" | "previous_answer" => Some("context"),
        "answer" | "response" => Some("answer"),
        _ => None,
    }
}
#[derive(Clone)]
struct Field {
    role: String,
    range: [usize; 2],
}
struct SystemPlan {
    id: String,
    kind: String,
    criteria: Value,
    range: Option<[usize; 2]>,
    target: Value,
}
fn system_state(state: &Value) -> Result<(Vec<Value>, HashMap<String, Field>)> {
    let mut state = state.clone();
    if let Some(s) = state.as_str() {
        if s.trim().starts_with('{') {
            if let Ok(obj) = serde_json::from_str::<Value>(s) {
                if obj.is_object() {
                    state = obj;
                }
            }
        }
    }
    let mut fields = HashMap::new();
    if let Some(s) = state.as_str() {
        if s.chars().all(python_space) {
            return invalid(
                vec![json!("state")],
                "state must not be empty or whitespace",
                "value_error",
            );
        }
        fields.insert(
            "request".into(),
            Field {
                role: "user".into(),
                range: [0, s.chars().count()],
            },
        );
        return Ok((vec![json!({"type":"user","text":s})], fields));
    }
    if state.is_array() {
        let s = canonical(&state)?;
        fields.insert(
            "request".into(),
            Field {
                role: "user".into(),
                range: [0, s.chars().count()],
            },
        );
        return Ok((vec![json!({"type":"user","text":s})], fields));
    }
    let object = match state.as_object() {
        Some(o) if !o.is_empty() => o,
        _ => {
            return invalid(
                vec![json!("state")],
                "state must be a non-empty string, JSON object or JSON array",
                "value_error",
            )
        }
    };
    let mut grouped: HashMap<&str, Vec<(String, String, bool)>> = HashMap::new();
    for (key, value) in object {
        let lower = key.to_lowercase();
        let known = state_role(&lower);
        grouped
            .entry(known.unwrap_or("context"))
            .or_default()
            .push((key.clone(), content(value)?, known.is_some()));
    }
    let mut parts = vec![];
    for role in ["user", "context", "answer"] {
        let items = match grouped.get(role) {
            Some(v) => v,
            None => continue,
        };
        let mut s = String::new();
        if items.len() == 1 && items[0].2 {
            let (key, t, _) = &items[0];
            s = t.clone();
            fields.insert(
                key.clone(),
                Field {
                    role: role.into(),
                    range: [0, t.chars().count()],
                },
            );
        } else {
            for (key, t, _) in items {
                if !s.is_empty() {
                    s += "\n\n";
                }
                let head = format!("{key}:\n");
                let start = s.chars().count() + head.chars().count();
                fields.insert(
                    key.clone(),
                    Field {
                        role: role.into(),
                        range: [start, start + t.chars().count()],
                    },
                );
                s += &head;
                s += t;
            }
        }
        if s.chars().all(python_space) {
            return invalid(
                vec![json!("state")],
                format!("the {role} part of the state is empty"),
                "value_error",
            );
        }
        parts.push(json!({"type":role,"text":s}));
    }
    Ok((parts, fields))
}
fn python_list(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|v| format!("'{}'", v.replace('\\', "\\\\").replace('\'', "\\'")))
            .collect::<Vec<_>>()
            .join(", ")
    )
}
fn resolve_over(
    over: &Value,
    fields: &HashMap<String, Field>,
    ordered_roles: &[String],
    location: &[Value],
) -> Result<(Value, Option<[usize; 2]>)> {
    let items = if let Some(a) = over.as_array() {
        a.clone()
    } else {
        vec![over.clone()]
    };
    let mut out = HashSet::new();
    let mut range = None;
    let mut loc = location.to_vec();
    loc.push(json!("over"));
    for item in items {
        let name = match item.as_str() {
            Some(v) => v,
            None => {
                return invalid(
                    loc,
                    "over must be a state key, a part name or a list of them",
                    "value_error",
                )
            }
        };
        let role = if let Some(f) = fields.get(name) {
            if !over.is_array() {
                range = Some(f.range);
            }
            f.role.clone()
        } else if ordered_roles.iter().any(|r| r == name) {
            name.into()
        } else if let Some(role) = state_role(name).filter(|r| ordered_roles.iter().any(|v| v == r))
        {
            role.into()
        } else {
            let mut keys = fields.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            return invalid(
                loc,
                format!(
                    "'{name}' is not a field of the state (fields: {}, parts: {})",
                    python_list(&keys),
                    python_list(ordered_roles)
                ),
                "value_error",
            );
        };
        out.insert(role);
    }
    if out.is_empty() {
        return invalid(loc, "over must not be empty", "value_error");
    }
    let roles = ordered_roles
        .iter()
        .filter(|r| out.contains(*r))
        .cloned()
        .collect::<Vec<_>>();
    Ok((
        if roles.len() == 1 {
            json!(roles[0])
        } else {
            json!(roles)
        },
        range,
    ))
}
fn system_request(state: &Value, questions: &Value) -> Result<(Value, Vec<SystemPlan>)> {
    let (parts, fields) = system_state(state)?;
    let roles = parts
        .iter()
        .map(|p| text(&p["type"]).map(str::to_string))
        .collect::<Result<Vec<_>>>()?;
    let qs = match questions.as_object() {
        Some(o) if !o.is_empty() => o,
        _ => {
            return invalid(
                vec![json!("questions")],
                "questions must be a non-empty object of named questions",
                "value_error",
            )
        }
    };
    let mut plans = vec![];
    let mut request_questions = vec![];
    for (id, q) in qs {
        let loc = vec![json!("questions"), json!(id)];
        if id.chars().all(python_space) {
            return invalid(
                vec![json!("questions")],
                "question IDs must not be empty or whitespace",
                "value_error",
            );
        }
        let object = match q.as_object() {
            Some(o) => o,
            None => {
                return invalid(
                    vec![json!("questions")],
                    "a question must be an object with a type",
                    "value_error",
                )
            }
        };
        let kind = q["type"].as_str().unwrap_or("");
        if !["noul", "choice", "score", "span", "set"].contains(&kind) {
            let mut l = loc.clone();
            l.push(json!("type"));
            return invalid(
                l,
                "type must be one of ['noul', 'choice', 'score', 'span', 'set']",
                "literal_error",
            );
        }
        let allowed = ["type", "instructions", "criteria", "over"];
        let mut extra = object
            .keys()
            .filter(|k| {
                !allowed.contains(&k.as_str())
                    && !(k.as_str() == "threshold" && ["span", "set"].contains(&kind))
            })
            .collect::<Vec<_>>();
        extra.sort();
        if let Some(k) = extra.first() {
            let mut l = loc.clone();
            l.push(json!(k));
            return invalid(l, "Extra inputs are not permitted", "extra_forbidden");
        }
        if !nonblank(&q["instructions"]) {
            let mut l = loc.clone();
            l.push(json!("instructions"));
            return invalid(
                l,
                "instructions must not be empty or whitespace",
                "value_error",
            );
        }
        let instructions = if q["instructions"].is_null() {
            String::new()
        } else {
            content(&q["instructions"])?
        };
        let (target, range) = if !q["over"].is_null() {
            resolve_over(&q["over"], &fields, &roles, &loc)?
        } else if kind == "span" {
            (
                json!(if roles.iter().any(|r| r == "answer") {
                    "answer"
                } else if roles.iter().any(|r| r == "user") {
                    "user"
                } else {
                    roles[0].as_str()
                }),
                None,
            )
        } else {
            (
                if roles.len() == 1 {
                    json!(roles[0])
                } else {
                    json!(roles)
                },
                None,
            )
        };
        let criteria = &q["criteria"];
        let mut generic = json!({"id":id,"type":kind,"text":instructions,"target_part":target});
        if kind == "noul" {
            if !criteria.is_null()
                && (!criteria.is_object()
                    || criteria
                        .as_object()
                        .context("criteria object")?
                        .keys()
                        .any(|k| !["true", "false"].contains(&k.as_str())))
            {
                let mut l = loc.clone();
                l.push(json!("criteria"));
                return invalid(
                    l,
                    "noul criteria is an object with optional true / false",
                    "value_error",
                );
            }
            for key in ["true", "false"] {
                if !nonblank(&criteria[key]) {
                    let mut l = loc.clone();
                    l.extend([json!("criteria"), json!(key)]);
                    return invalid(
                        l,
                        "Noul criterion must not be empty or whitespace",
                        "value_error",
                    );
                }
            }
            let no = if criteria["false"].is_null() {
                "No. The statement or question is not satisfied.".into()
            } else {
                content(&criteria["false"])?
            };
            let yes = if criteria["true"].is_null() {
                "Yes. The statement or question is satisfied.".into()
            } else {
                content(&criteria["true"])?
            };
            generic["type"] = json!("choice");
            generic["options"] =
                json!([{"name":"no","description":no},{"name":"yes","description":yes}]);
        } else if ["choice", "set", "span"].contains(&kind) {
            let minimum = if kind == "choice" { 2 } else { 1 };
            let object = match criteria.as_object() {
                Some(o) if (minimum..=255).contains(&o.len()) => o,
                _ => {
                    let mut l = loc.clone();
                    l.push(json!("criteria"));
                    return invalid(
                        l,
                        format!("{kind} criteria must be an object with {minimum} to 255 entries"),
                        "value_error",
                    );
                }
            };
            let mut options = vec![];
            for (name, value) in object {
                if name.chars().all(python_space) || !nonblank(value) {
                    let mut l = loc.clone();
                    l.extend([json!("criteria"), json!(name)]);
                    return invalid(
                        l,
                        "option names and descriptions must not be empty or whitespace",
                        "value_error",
                    );
                }
                options.push(json!({"name":name,"description":if value.is_null(){String::new()}else{content(value)?}}));
            }
            generic[if kind == "span" { "labels" } else { "options" }] = json!(options);
        } else {
            let levels = match criteria.as_array() {
                Some(v) if (2..=10).contains(&v.len()) => v,
                _ => {
                    let mut l = loc.clone();
                    l.push(json!("criteria"));
                    return invalid(
                        l,
                        "score criteria must be a list of 2 to 10 levels",
                        "value_error",
                    );
                }
            };
            let mut names = vec![];
            for (i, value) in levels.iter().enumerate() {
                if value.is_null() || !nonblank(value) {
                    let mut l = loc.clone();
                    l.extend([json!("criteria"), json!(i)]);
                    return invalid(
                        l,
                        "Score criterion must not be empty or whitespace",
                        "value_error",
                    );
                }
                names.push(content(value)?);
            }
            if names.iter().collect::<HashSet<_>>().len() != names.len() {
                let mut l = loc.clone();
                l.push(json!("criteria"));
                return invalid(l, "score levels must be distinct", "value_error");
            }
            generic["options"] = json!(names);
        }
        if !q["threshold"].is_null() {
            let t = q["threshold"].as_f64();
            if !t.is_some_and(|v| (0.0..=1.0).contains(&v)) {
                let mut l = loc.clone();
                l.push(json!("threshold"));
                return invalid(l, "threshold must be a number in [0, 1]", "value_error");
            }
            generic["threshold"] = q["threshold"].clone();
        }
        plans.push(SystemPlan {
            id: id.clone(),
            kind: kind.into(),
            criteria: criteria.clone(),
            range,
            target,
        });
        request_questions.push(generic);
    }
    for plan in &plans {
        if plan.kind == "set" {
            for label in plan.criteria.as_object().context("criteria object")?.keys() {
                let id = format!("{}.{label}", plan.id);
                if qs.contains_key(&id) {
                    return invalid(
                        vec![json!("questions"), json!(id)],
                        "collides with the Noul view of a set label",
                        "value_error",
                    );
                }
            }
        }
    }
    Ok((json!({"parts":parts,"questions":request_questions}), plans))
}

impl Frontend {
    pub fn system_one<F>(
        &self,
        state: &Value,
        questions: &Value,
        model: Option<&str>,
        mut execute: F,
    ) -> Result<Value>
    where
        F: FnMut(&Inputs) -> Result<Logits>,
    {
        let (request, plans) = system_request(state, questions)?;
        let limits = plans
            .iter()
            .filter(|p| p.kind == "span")
            .filter_map(|p| p.range.map(|r| (p.id.clone(), r)))
            .collect::<HashMap<_, _>>();
        let prediction = match self.predict_inner(&request, &limits, &mut execute) {
            Ok(p) => p,
            Err(e) if e.to_string().contains("does not fit") => {
                return invalid(
                    vec![json!("questions")],
                    format!(
                        "the questions alone do not fit the {}-token window",
                        self.config["max_length"]
                    ),
                    "too_long",
                )
            }
            Err(e) => return Err(e),
        };
        let mut answers = serde_json::Map::new();
        let mut spans = serde_json::Map::new();
        let mut sets = serde_json::Map::new();
        let mut thresholds = serde_json::Map::new();
        let response = array(&prediction.response["answers"])?;
        for plan in plans {
            let answer = response
                .iter()
                .find(|a| a["id"] == plan.id)
                .context("system answer")?;
            let probabilities = &answer["probabilities"];
            match plan.kind.as_str() {
                "noul" => {
                    answers.insert(plan.id, json!({"type":"noul","noul":probabilities["yes"]}));
                }
                "choice" => {
                    let mut pr = serde_json::Map::new();
                    let mut values = vec![];
                    for name in plan.criteria.as_object().context("criteria object")?.keys() {
                        let p = probabilities[name].as_f64().context("choice probability")?;
                        pr.insert(name.clone(), json!(p));
                        values.push(p);
                    }
                    values.sort_by(|a, b| b.total_cmp(a));
                    let confidence = if values.len() == 1 {
                        1.
                    } else {
                        (values[0] - values[1]).clamp(0., 1.)
                    };
                    answers.insert(plan.id,json!({"type":"choice","choice":answer["answer"],"confidence":confidence,"probabilities":pr}));
                }
                "score" => {
                    let levels = array(&plan.criteria)?;
                    let p = levels
                        .iter()
                        .map(|level| {
                            let name = content(level)?;
                            probabilities[name].as_f64().context("score probability")
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let mean = p.iter().enumerate().map(|(i, p)| i as f64 * p).sum::<f64>();
                    let variance = p
                        .iter()
                        .enumerate()
                        .map(|(i, p)| p * (i as f64 - mean).powi(2))
                        .sum::<f64>();
                    let n = p.len();
                    let confidence = if n == 1 {
                        1.
                    } else {
                        (1. - variance / ((n * n - 1) as f64 / 12.)).clamp(0., 1.)
                    };
                    let legend = levels
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i.to_string(), v.clone()))
                        .collect::<serde_json::Map<_, _>>();
                    let pr = p
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| (i.to_string(), json!(v)))
                        .collect::<serde_json::Map<_, _>>();
                    answers.insert(plan.id,json!({"type":"score","score":mean,"confidence":confidence,"legend":legend,"probabilities":pr}));
                }
                "set" => {
                    let mut pr = serde_json::Map::new();
                    for name in plan.criteria.as_object().context("criteria object")?.keys() {
                        let value = probabilities[name].clone();
                        pr.insert(name.clone(), value.clone());
                        answers.insert(
                            format!("{}.{name}", plan.id),
                            json!({"type":"noul","noul":value}),
                        );
                    }
                    thresholds.insert(plan.id.clone(), answer["threshold"].clone());
                    sets.insert(
                        plan.id,
                        json!({"selected":answer["answer"],"probabilities":pr}),
                    );
                }
                "span" => {
                    let mut out = vec![];
                    for span in array(&answer["spans"])? {
                        let mut value = span.clone();
                        if let Some([lo, hi]) = plan.range {
                            let part = array(&request["parts"])?
                                .iter()
                                .find(|p| p["type"] == plan.target)
                                .context("span range part")?;
                            let source = text(&part["text"])?;
                            let start = span["start"].as_u64().context("span start")? as usize;
                            let end = span["end"].as_u64().context("span end")? as usize;
                            let (start, end) = trim(source, start.max(lo), end.min(hi))?;
                            if end <= start {
                                continue;
                            }
                            let bytes = chars_to_bytes(source);
                            value["start"] = json!(start - lo);
                            value["end"] = json!(end - lo);
                            value["text"] = json!(&source[bytes[start]..bytes[end]]);
                        }
                        out.push(value);
                    }
                    thresholds.insert(plan.id.clone(), answer["threshold"].clone());
                    answers.insert(plan.id.clone(),json!({"type":"noul","noul":prediction.span_max.get(&plan.id).copied().unwrap_or(0.)}));
                    spans.insert(plan.id, json!(out));
                }
                _ => invalid_shape!("unsupported SystemOne plan"),
            }
        }
        let mut out = json!({"model":model.filter(|s|!s.is_empty()).unwrap_or("vllm-sr/Vela-2.0-0.3B"),"answers":answers,"usage":{"input_tokens":prediction.input_tokens,"output_tokens":0}});
        if !spans.is_empty() {
            out["spans"] = json!(spans);
        }
        if !sets.is_empty() {
            out["sets"] = json!(sets);
        }
        if !thresholds.is_empty() {
            out["thresholds"] = json!(thresholds);
        }
        Ok(out)
    }
    pub fn system_one_request<F>(&self, body: &Value, execute: F) -> Result<Value>
    where
        F: FnMut(&Inputs) -> Result<Logits>,
    {
        let object = match body.as_object() {
            Some(o) => o,
            None => {
                return invalid(
                    vec![],
                    "the request body must be a JSON object",
                    "value_error",
                )
            }
        };
        for key in ["state", "model", "questions"] {
            if !object.contains_key(key) {
                return invalid(vec![json!(key)], "Field required", "missing");
            }
        }
        let mut extra = object
            .keys()
            .filter(|k| !["state", "model", "questions"].contains(&k.as_str()))
            .collect::<Vec<_>>();
        extra.sort();
        if let Some(key) = extra.first() {
            return invalid(
                vec![json!(key)],
                "Extra inputs are not permitted",
                "extra_forbidden",
            );
        }
        let model = match body["model"].as_str() {
            Some(s) if !s.chars().all(python_space) => s,
            _ => {
                return invalid(
                    vec![json!("model")],
                    "model must not be empty or whitespace",
                    "value_error",
                )
            }
        };
        self.system_one(&body["state"], &body["questions"], Some(model), execute)
    }
}

/// Canonical SystemOne preparation without running the model.
pub fn system_preparation(state: &Value, questions: &Value) -> Result<(Value, Value)> {
    let (req, plan) = system_request(state, questions)?;
    Ok((req,json!(plan.iter().map(|p|json!({"qid":p.id,"type":p.kind,"criteria":p.criteria,"range":p.range,"part":p.target})).collect::<Vec<_>>())))
}

impl Inputs {
    /// Check all indices before a Vela forward. Index arrays include dummy rows.
    pub fn validate(&self, max_seq: usize, vocab: usize) -> Result<(usize, usize)> {
        let b = self.input_ids.len();
        check_shape!(b > 0, "Vela empty batch");
        let s = self.input_ids[0].len();
        check_shape!(
            s > 0 && s <= max_seq && self.attention_mask.len() == b,
            "Vela shape [{b}, {s}], mask rows {}, max_seq {max_seq}",
            self.attention_mask.len()
        );
        for (i, (ids, mask)) in self.input_ids.iter().zip(&self.attention_mask).enumerate() {
            check_shape!(
                ids.len() == s && mask.len() == s,
                "Vela ragged row {i}: ids {}, mask {}, width {s}",
                ids.len(),
                mask.len()
            );
            check_shape!(
                ids.iter().all(|&v| (v as usize) < vocab)
                    && mask.iter().all(|&v| v <= 1)
                    && mask.contains(&1),
                "Vela invalid token or mask in row {i}, vocab {vocab}"
            );
        }
        check_shape!(
            !self.q_index.is_empty()
                && !self.opt_index.is_empty()
                && !self.unit_index.is_empty()
                && !self.ent_index.is_empty(),
            "Vela dummy index rows required"
        );
        for q in &self.q_index {
            check_shape!(
                q[0] < b && q[1] < s && q[2] <= q[3] && q[3] <= s,
                "Vela question index {q:?} outside [{b}, {s}]"
            );
        }
        for o in &self.opt_index {
            check_shape!(
                o[0] < b && o[1] < s && o[2] < self.q_index.len(),
                "Vela option index {o:?} outside [{b}, {s}] / {} questions",
                self.q_index.len()
            );
        }
        for rows in [&self.unit_index, &self.ent_index] {
            check_shape!(
                rows.iter().all(|v| v[0] < b && v[1] < s),
                "Vela span index outside [{b}, {s}]"
            );
        }
        rt_core::checked_size("Vela input tensor", &[b, s])?;
        Ok((b, s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_deserialized_plan_returns_error() {
        let plan:Plan=serde_json::from_value(json!({"batches":[{"jobs":[0],"inputs":{"input_ids":[[1]],"attention_mask":[[1]],"q_index":[[0,0,0,1]],"opt_index":[[0,0,0]],"unit_index":[[0,0]],"ent_index":[[0,0]]}}],"jobs":[],"rows":[],"tokens":[],"full_words":[]})).unwrap();
        let out = vec![Logits {
            opt_logits: vec![0.],
            span_logits: vec![vec![0.]],
        }];
        assert!(validate_plan(&plan, &out)
            .unwrap_err()
            .to_string()
            .contains("job index"));
    }
    #[test]
    fn canonical_large_integer_preserves_exact_digits() {
        let input =
            "{\"big\":18446744073709551617,\"negative\":-18446744073709551617,\"float\":1.0}";
        let value: Value = serde_json::from_str(input).unwrap();
        assert_eq!(
            canonical(&value).unwrap(),
            "{\"big\":18446744073709551617,\"float\":1.0,\"negative\":-18446744073709551617}"
        );
    }
    #[test]
    fn nonoverlapping_field_clips_are_empty_without_panicking() {
        assert_eq!(trim("John Smith\n\nfield: 김철수", 20, 5).unwrap(), (20, 5));
        assert_eq!(trim("value", 3, 3).unwrap(), (3, 3));
    }
    #[test]
    fn unicode_offsets_url_trimming_and_canonical_json_match_contract() {
        let s = "[https://example.com/a).] 김철수";
        let (a, b) = trim(s, 0, 24).unwrap();
        assert_eq!(
            s.chars().skip(a).take(b - a).collect::<String>(),
            "https://example.com/a"
        );
        assert_eq!(
            canonical(&json!({"z":1e-5,"a":1.0,"k":"김"})).unwrap(),
            "{\"a\":1.0,\"k\":\"김\",\"z\":1e-05}"
        );
    }
}

#[cfg(test)]
mod bad_input_tests {
    use super::*;
    fn inputs() -> Inputs {
        Inputs {
            input_ids: vec![vec![1, 2]],
            attention_mask: vec![vec![1, 1]],
            q_index: vec![[0, 0, 0, 2]],
            opt_index: vec![[0, 0, 0]],
            unit_index: vec![[0, 0]],
            ent_index: vec![[0, 0]],
        }
    }
    fn frontend() -> Frontend {
        let tok = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}"#;
        Frontend {
            config: json!({"max_length":32,"marker_ids":{},"bos_token_id":0,"eos_token_id":0,"pad_token_id":0}),
            calibration: json!({}),
            tokenizer: Tokenizer::from_bytes(tok).unwrap(),
            word: Regex::new(".").unwrap(),
            unit: Regex::new(".").unwrap(),
            cache: Mutex::new(HashMap::new()),
            batch_size: 8,
            token_budget: 64,
        }
    }
    #[test]
    fn inputs_reject_empty_ragged_and_invalid_indices() {
        assert_eq!(inputs().validate(2, 4).unwrap(), (1, 2));
        let mut r = inputs();
        r.input_ids.clear();
        assert!(r.validate(2, 4).is_err());
        let mut r = inputs();
        r.attention_mask.clear();
        assert!(r.validate(2, 4).is_err());
        let mut r = inputs();
        r.input_ids[0].clear();
        assert!(r.validate(2, 4).is_err());
        let mut r = inputs();
        r.attention_mask[0].pop();
        assert!(r.validate(2, 4).is_err());
        let mut r = inputs();
        r.input_ids[0][0] = 4;
        assert!(r.validate(2, 4).is_err());
        let mut r = inputs();
        r.attention_mask[0][0] = 2;
        assert!(r.validate(2, 4).is_err());
        let mut r = inputs();
        r.q_index[0][3] = usize::MAX;
        assert!(r.validate(2, 4).is_err());
        let mut r = inputs();
        r.opt_index[0][2] = 1;
        assert!(r.validate(2, 4).is_err());
        let mut r = inputs();
        r.unit_index[0][0] = 1;
        assert!(r.validate(2, 4).is_err());
        assert!(inputs().validate(1, 4).is_err());
    }
    #[test]
    fn invalid_frontend_inputs_return_errors() {
        let mut fe = frontend();
        assert!(fe.prepare(&[]).is_err());
        assert!(fe.prepare(&[json!({})]).is_err());
        assert!(fe.prepare(&[json!({"parts":[{"role":"user","text":"hello"}],"questions":[{"type":"choice","text":"q","over":"missing","options":{}}]})]).is_err());
        fe.batch_size = 0;
        assert!(fe.prepare(&[json!({})]).is_err());
        assert!(fe
            .decode_spans(&[vec![0.5]], &[[1, usize::MAX]], &["x".into()], "x", 0.1)
            .is_err());
        assert!(fe
            .decode_spans(&[vec![]], &[[0, 1]], &["x".into()], "x", 0.1)
            .is_err());
        assert!(trim("x", 0, usize::MAX).is_err());
        assert!(matches!(
            invalid::<()>(vec![json!("questions")], "bad", "value_error"),
            Err(Error::Request(_))
        ));
    }
    #[test]
    fn malformed_files_return_errors_without_assets() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/round1-input-tests/vela-tokenizer");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("calibration.json"), "{}").unwrap();
        for (config, tokenizer) in [
            ("{", "{}"),
            ("[]", "{}"),
            (r#"{"marker_ids":{}}"#, "[]"),
            (r#"{"marker_ids":{}}"#, r#"{"added_tokens":[]}"#),
        ] {
            std::fs::write(dir.join("config.json"), config).unwrap();
            std::fs::write(dir.join("tokenizer.json"), tokenizer).unwrap();
            assert!(Frontend::load(&dir).is_err());
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
