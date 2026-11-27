# COLOSSUS reproduction inputs and research harness

This is the frozen experiment protocol and external research harness, not a new
application subsystem. It creates model data and experiment outputs only under
`D:\QML-COLOSSUS`. The measured source revision, binary/source hashes, environment,
cache identity, outcomes and limits are recorded in [the research report](PROJECT_COLOSSUS.md)
and [experiment 007](../experiments/exp-007-project-colossus.toml).

Use the existing workspace crates as path dependencies in an external Rust 2024
package named `colossus-experiment`, with its own empty `[workspace]`, and existing
`serde_json = "1"`. Dependencies are qml-core, qml-model, qml-format, qml-quant,
qml-forge, qml-runtime, qml-tokenizer and qml-behavior. Use the repository lockfile
and `CARGO_TARGET_DIR=D:\QML-COLOSSUS\scratch\build`; set TEMP/TMP to the dedicated
scratch directory. No application dependency is added. The driver expects the
pinned and fully SHA-256 verified source plus the local provenance records
before `prepare`; no model download is performed by this code.

Run `prepare`, `reference`, then `forge int3`, `evaluate int3` and the oracle for
`int3`; repeat with `int2` and `qmr200`. Run each in a fresh process, preserve
stdout/stderr, measure elapsed time and sampled process working set externally,
and retain all failures. Output creation refuses replacement. Existing cache
serialization, algorithms and ATLAS publication remain authoritative. The two
oracle self-test modes exercise known-correct and deliberately wrong controls;
they are not generated model outputs. The optional Python interpreter is only
an execution oracle for requested generated Python snippets. Tests are held out
from calibration, with no claim that their general concepts are absent from
model pretraining. All seven numerical reference sequences are plain text;
the separate coding tasks use ChatML. Numerical results do not replace task
correctness. Review pure generated snippets before using them outside this
controlled experiment.

## Frozen protocol

```json
{
  "schema": 2,
  "model": "Qwen/Qwen2.5-Coder-14B-Instruct",
  "revision": "aedcc2d42b622764e023cf882b6652e646b95671",
  "system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
  "temperature": 0,
  "seed": 42,
  "max_new_tokens": 384,
  "max_seq_len": 1024,
  "hessian_block": 64,
  "group_size": 128,
  "hadamard_block": 64,
  "kv_cache": "int8",
  "tasks": [
    {
      "id": "rust-1",
      "category": "rust_generation",
      "language": "rust",
      "prompt": "Return only Rust code defining pub fn runs(xs: &[i32]) -> Vec<(i32, usize)> that run-length encodes consecutive equal values in order. No main, imports, or external crates.",
      "tests": "fn main(){assert_eq!(runs(&[]),vec![]);assert_eq!(runs(&[1,1,2,1]),vec![(1,2),(2,1),(1,1)]);assert_eq!(runs(&[-1,-1,-1]),vec![(-1,3)]);assert_eq!(runs(&[3,4]),vec![(3,1),(4,1)]);}"
    },
    {
      "id": "rust-2",
      "category": "rust_generation",
      "language": "rust",
      "prompt": "Return only Rust code defining pub fn first_unique(s: &str) -> Option<char>. Return the first Unicode character occurring exactly once, in original order; None if none. No main or external crates.",
      "tests": "fn main(){assert_eq!(first_unique(\"\"),None);assert_eq!(first_unique(\"aabbcdd\"),Some('c'));assert_eq!(first_unique(\"é🙂éx🙂\"),Some('x'));assert_eq!(first_unique(\"abac\"),Some('b'));assert_eq!(first_unique(\"aa\"),None);}"
    },
    {
      "id": "python-1",
      "category": "python_generation",
      "language": "python",
      "prompt": "Return only Python code defining merge_intervals(intervals). Inputs are pairs (start,end), start<=end. Return a sorted list of tuples merging overlapping or touching closed intervals. Do not mutate input. No imports or IO.",
      "tests": "assert merge_intervals([])==[]\na=[(5,7),(1,3),(3,5),(10,10)]; assert merge_intervals(a)==[(1,7),(10,10)]; assert a==[(5,7),(1,3),(3,5),(10,10)]\nassert merge_intervals([(-3,-1),(-2,0),(2,3)])==[(-3,0),(2,3)]\nassert merge_intervals([(1,5),(2,3),(1,5)])==[(1,5)]"
    },
    {
      "id": "python-2",
      "category": "python_generation",
      "language": "python",
      "prompt": "Return only Python code defining rotate(xs,k). Return a new list rotated right by k; negative k rotates left. Handle empty input without error and do not mutate xs. No imports or IO.",
      "tests": "assert rotate([],10)==[]\na=[1,2,3,4]; assert rotate(a,1)==[4,1,2,3]; assert a==[1,2,3,4]\nassert rotate(a,-1)==[2,3,4,1]\nassert rotate(a,8)==a\nassert rotate([9],-5)==[9]"
    },
    {
      "id": "js-1",
      "category": "javascript_generation",
      "language": "javascript",
      "prompt": "Return only JavaScript code defining function uniqueStable(xs). Return a new array with duplicates removed, retaining first occurrence order. Inputs contain finite numbers and strings. Treat number 1 and string '1' as distinct. Do not mutate xs. No imports or IO.",
      "tests": "const a=[1,'1',1,2,'1',3]; if(JSON.stringify(uniqueStable(a))!==JSON.stringify([1,'1',2,3]))throw Error('mixed');if(a.length!==6)throw Error('mutation');if(uniqueStable([]).length)throw Error('empty');if(JSON.stringify(uniqueStable(['a','a','b']))!==JSON.stringify(['a','b']))throw Error('strings');"
    },
    {
      "id": "js-2",
      "category": "javascript_generation",
      "language": "javascript",
      "prompt": "Return only JavaScript code defining function lowerBound(xs,target), returning the first index i where sorted numeric xs[i]>=target, or xs.length if none. Use O(log n) time. No imports or IO.",
      "tests": "for(const [a,t,e] of [[[],1,0],[[1,2,2,2,4],2,1],[[1,2],3,2],[[1,2],0,0],[[-4,-2,0],-3,1]])if(lowerBound(a,t)!==e)throw Error('index');"
    },
    {
      "id": "debug-1",
      "category": "debugging",
      "language": "rust",
      "prompt": "Fix this Rust function. Return only corrected pub fn last_index(xs: &[i32], value: i32) -> Option<usize>, no main. It must return the last matching index, including empty input safely. Broken code: fn last_index(xs: &[i32],value:i32)->Option<usize>{ for i in (0..=xs.len()).rev(){if xs[i]==value{return Some(i)}} None }",
      "tests": "fn main(){assert_eq!(last_index(&[],1),None);assert_eq!(last_index(&[2,1,2],2),Some(2));assert_eq!(last_index(&[2],2),Some(0));assert_eq!(last_index(&[1,3],2),None);}"
    },
    {
      "id": "debug-2",
      "category": "debugging",
      "language": "python",
      "prompt": "Fix Python's mutable-default bug while preserving explicit-list behavior. Return only code defining append_item(x, items=None), returning the list after appending x. Each omitted-items call gets a fresh list; an explicitly supplied list must be mutated and returned. Broken code: def append_item(x, items=[]): items.append(x); return items. No imports or IO.",
      "tests": "a=append_item(1); b=append_item(2); assert a==[1] and b==[2] and a is not b\nc=[]; assert append_item(3,c) is c and c==[3]\nassert append_item(4,c)==[3,4]"
    },
    {
      "id": "reason-1",
      "category": "algorithmic_reasoning",
      "language": "json",
      "prompt": "An undirected weighted graph has edges A-B:4,A-C:1,C-B:2,B-D:1,C-D:7. Find a shortest A-to-D path and its weight. Return only JSON with keys path (array of vertex strings) and weight (integer).",
      "expected": {
        "path": [
          "A",
          "C",
          "B",
          "D"
        ],
        "weight": 4
      }
    },
    {
      "id": "reason-2",
      "category": "algorithmic_reasoning",
      "language": "json",
      "prompt": "Compute the Levenshtein distance from kitten to sitting, with insertion, deletion, substitution each cost 1. Return only JSON {\"distance\":integer,\"dp_time\":\"...\"}, with dp_time the standard dynamic-programming time complexity for lengths m,n, using exactly O(m*n).",
      "expected": {
        "distance": 3,
        "dp_time": "O(m*n)"
      }
    },
    {
      "id": "explain-1",
      "category": "code_explanation",
      "language": "json",
      "prompt": "For Python code a=[[0]]*3; a[0].append(1), report final a and whether all three entries reference the same list. Return only JSON with keys value (nested arrays), shared (boolean), cause (exact string 'list repetition aliases the inner list' if shared, otherwise 'independent lists').",
      "expected": {
        "value": [
          [
            0,
            1
          ],
          [
            0,
            1
          ],
          [
            0,
            1
          ]
        ],
        "shared": true,
        "cause": "list repetition aliases the inner list"
      }
    },
    {
      "id": "explain-2",
      "category": "code_explanation",
      "language": "json",
      "prompt": "For JavaScript const a=[1,2,3]; const b=a.map(x=>x*2); b.pop(); report final a, b, and whether map mutates its input. Return only JSON with keys a, b, mutates (boolean).",
      "expected": {
        "a": [
          1,
          2,
          3
        ],
        "b": [
          2,
          4
        ],
        "mutates": false
      }
    },
    {
      "id": "instruction-1",
      "category": "instruction_following",
      "language": "json",
      "prompt": "Return only a JSON object with exactly two keys: names, containing the strings Rust, Python, JavaScript in alphabetical order; and count, containing their number. No Markdown or additional keys.",
      "expected": {
        "names": [
          "JavaScript",
          "Python",
          "Rust"
        ],
        "count": 3
      }
    },
    {
      "id": "instruction-2",
      "category": "instruction_following",
      "language": "text",
      "prompt": "Output exactly three lines, nothing else. Line 1: ALPHA. Line 2: the integer result of 7*8. Line 3: omega in lowercase.",
      "expected": "ALPHA\n56\nomega"
    }
  ],
  "calibration": [
    "Rust ownership moves values unless a type implements Copy. Borrowing gives access through references. An immutable reference allows reading and a mutable reference allows exclusive modification. Lifetimes describe reference validity.",
    "Python generators yield values lazily. A dictionary maps hashable keys to values. Context managers close resources reliably, including when an exception is raised. Sorting with a key computes ordering attributes.",
    "JavaScript promises represent asynchronous results. Await resumes when a promise settles. Array filter returns a new array of values matching a predicate. A closure retains access to lexical variables.",
    "A stack uses last in first out order. A queue uses first in first out order. Hash tables provide expected constant time lookup. Binary search trees compare keys and traverse left or right branches.",
    "Rust Result represents success or error. The question mark operator propagates an error to the caller. Pattern matching handles enum variants explicitly. Iterators can transform and collect items.",
    "Testing checks edge cases, normal cases, and malformed inputs. Deterministic tests avoid external network services. A regression test captures previously failing behavior and checks the expected contract.",
    "Dynamic programming reuses solutions to smaller subproblems. Memoization caches recursively computed results. Iterative tabulation fills a table in dependency order. Space can sometimes be reduced by retaining recent rows.",
    "Integer arithmetic may overflow. Checked addition returns an optional result. Bounds checks protect memory accesses. Input validation should occur before allocating large buffers or interpreting untrusted offsets.",
    "Rust code: parsing and checked aggregation.\nfn sum_fields(line: &str) -> Result<u64, String> {\n    line.split(',').try_fold(0u64, |total, field| {\n        let n = field.trim().parse::<u64>().map_err(|e| e.to_string())?;\n        total.checked_add(n).ok_or_else(|| \"overflow\".to_string())\n    })\n}\nThe parser trims fields and propagates errors without panicking.",
    "Rust code: enum state transition.\n#[derive(Clone, Debug, PartialEq)]\nenum State { Idle, Running(u64), Finished }\nfn advance(state: State, elapsed: u64) -> State {\n    match state {\n        State::Idle => State::Running(0),\n        State::Running(n) => match n.checked_add(elapsed) {\n            Some(next) if next < 100 => State::Running(next),\n            _ => State::Finished,\n        },\n        State::Finished => State::Finished,\n    }\n}\nPattern guards describe valid state transitions.",
    "Rust code: a small owned record.\nstruct Counter { name: String, value: usize }\nimpl Counter {\n    fn new(name: &str) -> Self { Self { name: name.to_owned(), value: 0 } }\n    fn increment(&mut self) { self.value = self.value.saturating_add(1); }\n    fn describe(&self) -> String { format!(\"{}: {}\", self.name, self.value) }\n}\nThe constructor owns its name, and the immutable method formats without modifying the counter.",
    "Python code: counting primes.\ndef primes_below(limit):\n    if limit <= 2:\n        return []\n    sieve = [True] * limit\n    sieve[0] = sieve[1] = False\n    for p in range(2, int(limit ** 0.5) + 1):\n        if sieve[p]:\n            for multiple in range(p * p, limit, p):\n                sieve[multiple] = False\n    return [n for n, prime in enumerate(sieve) if prime]\nA boolean sieve removes multiples starting at the square of each prime.",
    "Python code: aggregate records.\ndef department_totals(records):\n    totals = {}\n    for record in records:\n        department = record['department']\n        amount = record.get('amount', 0)\n        if not isinstance(amount, (int, float)):\n            raise ValueError('amount must be numeric')\n        totals[department] = totals.get(department, 0) + amount\n    return totals\nMissing amounts default to zero; invalid amounts produce a descriptive error.",
    "Python code: recursive expression trees.\ndef evaluate(node):\n    if isinstance(node, (int, float)):\n        return node\n    operation, left, right = node\n    a, b = evaluate(left), evaluate(right)\n    if operation == 'add':\n        return a + b\n    if operation == 'mul':\n        return a * b\n    raise ValueError('unknown operation')\nEach internal node combines recursively computed child values.",
    "JavaScript code: immutable record update.\nfunction updateById(records, id, patch) {\n  let found = false;\n  const result = records.map(record => {\n    if (record.id !== id) return record;\n    found = true;\n    return {...record, ...patch, id: record.id};\n  });\n  if (!found) throw new Error('unknown id');\n  return result;\n}\nObject spread creates a replacement record while preserving its identity field.",
    "JavaScript code: asynchronous batch requests.\nasync function collectResults(jobs, execute) {\n  const results = await Promise.all(jobs.map(async job => {\n    try {\n      return {id: job.id, ok: true, value: await execute(job)};\n    } catch (error) {\n      return {id: job.id, ok: false, error: String(error)};\n    }\n  }));\n  return results;\n}\nThe result array retains input order and records each failure explicitly.",
    "JavaScript code: recursive tree summary.\nfunction summarize(node) {\n  if (node == null) return {count: 0, total: 0};\n  const left = summarize(node.left);\n  const right = summarize(node.right);\n  return {\n    count: 1 + left.count + right.count,\n    total: node.value + left.total + right.total\n  };\n}\nThe base case handles absent children, while each branch contributes exactly once."
  ],
  "numerical_evaluation": [
    "Rust: fn square(x: i32) -> i32 { x * x }\nThe function returns the square.",
    "Python: def maximum(a, b): return a if a > b else b\nThis chooses the larger value.",
    "JavaScript: const sum = xs.reduce((a,b)=>a+b,0);\nAn empty array produces zero.",
    "To debug a failing loop, inspect its bounds and termination condition before changing the algorithm.",
    "Breadth first search visits vertices in distance order using a queue.",
    "A pure function returns a result without modifying external state.",
    "Follow the requested output format exactly and avoid adding unrelated commentary."
  ],
  "interpretation": "Fourteen small custom tasks, not an established benchmark. Functional checks do not prove complexity or general coding capability. Numerical fidelity is separate. No measured BF16 task-retention claim without original autoregressive baseline."
}
```

## Rust driver

```rust
#![forbid(unsafe_code)]
use std::{error::Error,fs::{self,File,OpenOptions},io::Write,path::{Path,PathBuf},sync::Arc,time::Instant};
use qml_runtime::{TransformerConfig,QCoreModel,QCoreBackend,KvCacheMode};
use qml_quant::{ModelCalibrationProfile,ResonanceProfile,BehaviorImportanceMap};
use qml_behavior::{CapabilityExample,CapabilityCategory,ReferenceSuiteCache};
use serde_json::{Value,json};
type Result<T> = std::result::Result<T,Box<dyn Error>>;
const ROOT:&str="D:/QML-COLOSSUS";
const MODEL:&str="Qwen/Qwen2.5-Coder-14B-Instruct";
fn path(s:&str)->PathBuf{Path::new(ROOT).join(s)}
fn write_json(p:&Path,v:&Value)->Result<()>{let mut f=OpenOptions::new().write(true).create_new(true).open(p)?;serde_json::to_writer_pretty(&mut f,v)?;f.write_all(b"\n")?;f.sync_all()?;Ok(())}
fn read_json(p:&Path)->Result<Value>{Ok(serde_json::from_reader(File::open(p)?)?)}
fn tokens(t:&qml_tokenizer::BpeTokenizer,s:&str)->Vec<usize>{t.encode(s).into_iter().map(|n|n as usize).collect()}
fn config()->Result<TransformerConfig>{let mut c=TransformerConfig::from_hf_config_bytes(&fs::read(path("original-model/config.json"))?)?;c.max_seq_len=1024;c.validate()?;Ok(c)}
fn identity()->Result<String>{Ok(read_json(&path("experiment-results/prepared.json"))?["identity"].as_str().ok_or("identity missing")?.into())}
fn part_id(id:&str,name:&str)->String{qml_core::sha256_hex(format!("{id}:{name}").as_bytes())}
fn main()->Result<()>{
 let args:Vec<String>=std::env::args().collect();let cmd=args.get(1).ok_or("command required")?;
 let protocol=read_json(&path("experiment-results/protocol.json"))?;let cfg=config()?;
 let tok=qml_tokenizer::BpeTokenizer::from_file(path("original-model/tokenizer.json"))?;
 match cmd.as_str(){
 "prepare"=>{
   let provenance=read_json(&path("experiment-results/upstream-provenance.json"))?;
   if provenance["id"]!=MODEL||provenance["sha"]!=protocol["revision"]{return Err("source identity mismatch".into())}
   for f in provenance["siblings"].as_array().ok_or("inventory missing")?{
    let n=f["rfilename"].as_str().unwrap_or("");
    if n.ends_with(".safetensors"){if fs::metadata(path(&format!("original-model/{n}")))?.len()!=f["size"].as_u64().ok_or("size missing")?{return Err("checkpoint size mismatch".into())}}
   }
   let mut sha=qml_core::Sha256::new();sha.update(b"QML-COLOSSUS-v1");
   for p in ["experiment-results/protocol.json","experiment-results/upstream-provenance.json","original-model/config.json","original-model/tokenizer.json","original-model/tokenizer_config.json","original-model/model.safetensors.index.json","experiment-results/download-verification.json","experiment-results/runner-final-provenance.json"]{sha.update(p.as_bytes());sha.update(&fs::read(path(p))?);}
   sha.update(cfg.execution_identity()?.as_bytes());
   let source=qml_model::SafetensorsSource::open(&path("original-model"),&qml_model::PayloadBudget::default())?;
   let parameters=source.metadata().tensors.iter().try_fold(0u64,|sum,t|->Result<u64>{Ok(sum.checked_add(qml_model::tensor_element_count(t)?).ok_or("parameter arithmetic overflow")?)})?;
   if source.metadata().tensors.iter().any(|t|t.dtype!="BF16"){return Err("source contains non-BF16 weights".into())}
   write_json(&path("experiment-results/source-inspection.json"),&json!({"parameters":parameters,"metadata":source.metadata()}))?;
   let id=sha.finalize_hex();
   let system=protocol["system"].as_str().unwrap();
   let task_tokens:Vec<_>=protocol["tasks"].as_array().unwrap().iter().map(|t|tokens(&tok,&tok.format_chat_prompt(Some(system),t["prompt"].as_str().unwrap()))).collect();
   for ts in &task_tokens{if ts.len()+384>cfg.max_seq_len{return Err("task context exceeds configured limit".into())}}
   let calibration:Vec<_>=protocol["calibration"].as_array().unwrap().iter().map(|s|tokens(&tok,s.as_str().unwrap())).collect();
   let evaluation:Vec<_>=protocol["numerical_evaluation"].as_array().unwrap().iter().map(|s|tokens(&tok,s.as_str().unwrap())).collect();
   let result=json!({"identity":id,"parameters":parameters,"config":cfg,"task_tokens":task_tokens,"calibration":calibration,"evaluation":evaluation,"eos":tok.eos_token_id(),"calibration_token_count":calibration.iter().map(Vec::len).sum::<usize>(),"evaluation_prediction_count":evaluation.iter().map(|s|s.len()-1).sum::<usize>()});
   write_json(&path("experiment-results/prepared.json"),&result)?;println!("{result}");
 },
 "reference"=>{
   let prep=read_json(&path("experiment-results/prepared.json"))?;let id=identity()?;
   if path("calibration/complete.json").exists(){return Err("calibration already complete; refusing replacement".into())}
   let cal:Vec<Vec<usize>>=serde_json::from_value(prep["calibration"].clone())?;
   let eval:Vec<Vec<usize>>=serde_json::from_value(prep["evaluation"].clone())?;
   let result=qml_runtime::stream_calibrate_and_reference(&path("original-model"),MODEL,&cfg,&cal,&eval,64,true,&qml_model::PayloadBudget::default(),|done,total,stage|{eprintln!("{done}/{total} {stage}");})?;
   let memory=result.calibration.memory_bytes();
   let meta=ModelCalibrationProfile{layers:Default::default(),..result.calibration.clone()};
   let mut names=Vec::new();
   for(name,stats)in result.calibration.layers{
    let mut shard=meta.clone();shard.layers.insert(name.clone(),stats);
    let file=format!("{}.qcal",name.replace('.',"_"));
    shard.save_verified(&path(&format!("calibration/{file}")),&part_id(&id,&file))?;names.push(file);
   }
   for(i,logits)in result.eval_logits.into_iter().enumerate(){
    let example=CapabilityExample{category:CapabilityCategory::Coding,label:format!("numerical-{i}"),tokens:eval[i].clone()};
    let cache=ReferenceSuiteCache::from_precomputed(vec![example],vec![logits]);
    let file=format!("numerical-{i}.qref");cache.save_verified(&path(&format!("reference-evaluation/{file}")),&part_id(&id,&file))?;
   }
   let report=json!({"identity":id,"cache_shards":names,"profile_memory_bytes":memory,"streaming_elapsed_secs":result.elapsed_secs,"used_gpu":result.used_gpu,"peak_layer_vram_bytes":result.peak_layer_vram_bytes,"note":"Layer weight accounting, not total process or device peak"});
   write_json(&path("calibration/complete.json"),&report)?;println!("{report}");
 },
 "forge"=>{
   let name=args.get(2).ok_or("representation required")?;let id=identity()?;
   let complete=read_json(&path("calibration/complete.json"))?;if complete["identity"]!=id{return Err("calibration identity mismatch".into())}
   let mut cal=ModelCalibrationProfile::default();
   for file in complete["cache_shards"].as_array().unwrap(){let file=file.as_str().unwrap();let shard=ModelCalibrationProfile::load_verified(&path(&format!("calibration/{file}")),&part_id(&id,file))?;cal.model_id=shard.model_id;cal.calibration_tokens=shard.calibration_tokens;cal.block_size=shard.block_size;cal.layers.extend(shard.layers);}
   let strategy=match name.as_str(){
    "int3"=>qml_forge::TransformStrategy::Int3Hadamard{group_size:128,hadamard_block:64},
    "int2"=>qml_forge::TransformStrategy::Int2Hadamard{group_size:128,hadamard_block:64},
    "qmr200"=>qml_forge::TransformStrategy::ResonanceMapB{profile:ResonanceProfile::Qmr200,include_vq:false,include_echo:true,behavior_weights:BehaviorImportanceMap::default_priors()},
    _=>return Err("unauthorized representation".into())
   };
   let control=qml_forge::ForgeControl{on_progress:Some(Arc::new(|d,t,s|eprintln!("{d}/{t} {s}"))),..Default::default()};
   let options=qml_format::DiskBuildOptions{scratch_directory:path("scratch"),..Default::default()};
   let start=Instant::now();
   let report=qml_forge::forge_path_to_file_atomic_with_options(&path("original-model"),&path(&format!("compressed-qmf/{name}.qmf")),MODEL,"Apache-2.0","qwen2",&strategy,&qml_model::PayloadBudget::default(),42,Some(&cal),&control,&options,false)?;
   let result=json!({"identity":id,"representation":name,"elapsed_secs":start.elapsed().as_secs_f64(),"report":report});
   write_json(&path(&format!("experiment-results/{name}-forge.json")),&result)?;println!("{result}");
 },
 "evaluate"=>{
   let name=args.get(2).ok_or("representation required")?;let id=identity()?;
   let prep=read_json(&path("experiment-results/prepared.json"))?;
   let start=Instant::now();
   let mut reader=qml_format::QmfFileReader::open(File::open(path(&format!("compressed-qmf/{name}.qmf")))?,&qml_format::QmfBudget::default())?;
   let artifact_sha256=reader.verify_integrity_streaming()?;
   let forged=read_json(&path(&format!("experiment-results/{name}-forge.json")))?;
   if forged["identity"]!=id||forged["report"]["qmf_sha256"]!=artifact_sha256{return Err("QMF provenance or integrity mismatch".into())}
   let integrity_secs=start.elapsed().as_secs_f64();
   let start=Instant::now();
   let model=QCoreModel::from_qmf_reader_with_backend(&mut reader,cfg,QCoreBackend::GpuFull)?;
   let load_secs=start.elapsed().as_secs_f64();let stats=model.backend_stats();
   write_json(&path(&format!("experiment-results/{name}-backend.json")),&json!({"identity":id,"artifact_sha256":artifact_sha256,"integrity_secs":integrity_secs,"load_secs":load_secs,"backend":stats}))?;
   for i in 0..7{
    let eval:Vec<Vec<usize>>=serde_json::from_value(prep["evaluation"].clone())?;
    let example=CapabilityExample{category:CapabilityCategory::Coding,label:format!("numerical-{i}"),tokens:eval[i].clone()};
    let file=format!("numerical-{i}.qref");let cache=ReferenceSuiteCache::load_verified(&path(&format!("reference-evaluation/{file}")),vec![example],&part_id(&id,&file))?;
    let t=Instant::now();let report=qml_behavior::evaluate_capability_suite_with_cache(&cache,&model)?;
    write_json(&path(&format!("experiment-results/{name}-numerical-{i}.json")),&json!({"identity":id,"elapsed_secs":t.elapsed().as_secs_f64(),"report":report}))?;
   }
   let task_tokens:Vec<Vec<usize>>=serde_json::from_value(prep["task_tokens"].clone())?;
   for(task,prompt)in protocol["tasks"].as_array().unwrap().iter().zip(task_tokens){
    let task_id=task["id"].as_str().unwrap();let output=path(&format!("experiment-results/{name}-{task_id}.json"));if output.exists(){return Err("task output exists; refusing overwrite".into())}
    eprintln!("Generating {name} {task_id}");
    let generation=model.generate_with_kv_mode(&prompt,384,0.0,Some(151645),KvCacheMode::Int8,42,|_|true);
    let record=match generation{
     Ok(g)=>{let ids:Vec<u32>=g.generated_tokens.iter().map(|&n|n as u32).collect();json!({"identity":id,"task_id":task_id,"text":tok.decode(&ids,true),"eos_emitted":g.generated_tokens.last()==Some(&151645),"generation":g})},
     Err(e)=>json!({"identity":id,"task_id":task_id,"error":e.to_string()})
    };write_json(&output,&record)?;println!("{record}");
   }
 },
 _=>return Err("unknown command".into())
 };Ok(())
}
```

## Objective oracle

```javascript
// Fixed objective checks. No packages, services, or network are used.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const cp=require('node:child_process'),path=require('node:path'),crypto=require('node:crypto');
const root='D:/QML-COLOSSUS',rep=process.argv[2],selfTest=/^oracle-(negative-)?selftest(-v2)?$/.test(rep),negative=rep.startsWith('oracle-negative-');
if(!selfTest&&!['int3','int2','qmr200'].includes(rep))throw Error('representation');
const protocol=JSON.parse(fs.readFileSync(path.join(root,'experiment-results/protocol.json'),'utf8'));
const prepared=selfTest?{identity:'oracle-selftest-not-model-evidence'}:JSON.parse(fs.readFileSync(path.join(root,'experiment-results/prepared.json'),'utf8'));
const outdir=path.join(root,'experiment-results',rep+'-objective-tests');
fs.mkdirSync(outdir);
const results=[];
function strictText(text){return text.replace(/\r\n/g,'\n').replace(/\n$/,'');}
if(selfTest){
 assert.equal(strictText('ALPHA\n56\nomega\n'),'ALPHA\n56\nomega');
 assert.notEqual(strictText('\nALPHA\n56\nomega'),'ALPHA\n56\nomega');
 assert.notEqual(strictText(' ALPHA\n56\nomega '),'ALPHA\n56\nomega');
}
const controls={
 'rust-1':'pub fn runs(xs:&[i32])->Vec<(i32,usize)>{let mut r:Vec<(i32,usize)>=Vec::new();for &x in xs{if let Some(p)=r.last_mut(){if p.0==x{p.1+=1;continue;}}r.push((x,1));}r}',
 'rust-2':'pub fn first_unique(s:&str)->Option<char>{s.chars().find(|&c|s.chars().filter(|&d|d==c).count()==1)}',
 'python-1':'def merge_intervals(intervals):\n r=[]\n for a,b in sorted(intervals):\n  if r and a<=r[-1][1]: r[-1]=(r[-1][0],max(b,r[-1][1]))\n  else: r.append((a,b))\n return r',
 'python-2':'def rotate(xs,k):\n if not xs: return []\n k%=len(xs)\n return xs[-k:]+xs[:-k] if k else list(xs)',
 'js-1':'function uniqueStable(xs){return [...new Set(xs)];}',
 'js-2':'function lowerBound(xs,t){let l=0,r=xs.length;while(l<r){let m=l+Math.floor((r-l)/2);if(xs[m]<t)l=m+1;else r=m;}return l;}',
 'debug-1':'pub fn last_index(xs:&[i32],v:i32)->Option<usize>{xs.iter().rposition(|&x|x==v)}',
 'debug-2':'def append_item(x,items=None):\n if items is None: items=[]\n items.append(x)\n return items'
};
function save(name,content){fs.writeFileSync(path.join(outdir,name),content,{flag:'wx'});}
function run(tool,args,timeout=15000){
 const r=cp.spawnSync(tool,args,{cwd:outdir,timeout,encoding:'utf8',windowsHide:true,maxBuffer:1024*1024,
 env:{...process.env,TEMP:path.join(root,'scratch'),TMP:path.join(root,'scratch'),PYTHONPYCACHEPREFIX:path.join(root,'scratch','python-cache')}});
 return {status:r.status,signal:r.signal,stdout:r.stdout??'',stderr:r.stderr??'',error:r.error?.message??null};
}
for(const task of protocol.tasks){
 const record=selfTest?{identity:prepared.identity,text:negative?({'rust':'fn wrong(){}','python':'def wrong(): return None','javascript':'function wrong(){}','json':'{}','text':'wrong'}[task.language]):controls[task.id]??(typeof task.expected==='string'?task.expected:JSON.stringify(task.expected))}:JSON.parse(fs.readFileSync(path.join(root,'experiment-results',rep+'-'+task.id+'.json'),'utf8'));
 assert.equal(record.identity,prepared.identity);
 let text=(record.text??'').trim(),format=true;
 const fence=text.match(/^\x60\x60\x60[^\n]*\n([\s\S]*?)\n\x60\x60\x60$/);
 if(fence){text=fence[1].trim();format=false;}
 const r={id:task.id,category:task.category,language:task.language,pass:false,format_compliant:format,
  eos_emitted:record.eos_emitted??false,generated_tokens:record.generation?.generated_tokens.length??0,
  elapsed_micros:record.generation?.elapsed_micros??null,tokens_per_second:record.generation?.tokens_per_second??null,
  output_sha256:crypto.createHash('sha256').update(record.text??'').digest('hex')};
 try{
  if(record.error)throw Error(record.error);
  if(task.language==='json'){
   assert.deepEqual(JSON.parse(text),task.expected);
   if(task.category==='instruction_following')assert.equal(format,true);
   r.pass=true;
  }else if(task.language==='text'){assert.equal(strictText(record.text??''),task.expected);r.pass=true;}
  else if(task.language==='javascript'){
   if(/\b(require|process|import|fetch|WebAssembly)\b/.test(text))throw Error('rejected external execution feature');
   save(task.id+'.js',text+'\n'+task.tests);
   const context=vm.createContext(Object.create(null),{codeGeneration:{strings:false,wasm:false}});
   vm.runInContext('"use strict";\n'+text+'\n'+task.tests,context,{timeout:2000});r.pass=true;
  }else if(task.language==='rust'){
   if(/\b(unsafe|extern|include|include_str|include_bytes|asm|global_asm|env|option_env|std\s*::\s*(fs|net|process|env)|Command|TcpStream|File)\b/.test(text))throw Error('rejected external execution feature');
   const file=path.join(outdir,task.id+'.rs'),exe=path.join(outdir,task.id+'.exe');
   save(task.id+'.rs',text+'\n'+task.tests);
   const compile=run('rustc.exe',['--edition=2024','--forbid','unsafe_code',file,'-o',exe],30000);
   save(task.id+'-compile.json',JSON.stringify(compile,null,2));
   if(compile.status!==0)throw Error('compilation failed');
   const test=run(exe,[],3000);save(task.id+'-run.json',JSON.stringify(test,null,2));
   if(test.status!==0)throw Error('unit assertions failed or timeout');r.pass=true;
  }else if(task.language==='python'){
   save(task.id+'.py',text+'\n'+task.tests);
   // This is a generated-code test oracle, not QML application code or orchestration.
   const guard="import ast, pathlib, sys\np=pathlib.Path(sys.argv[1]); s=p.read_text(encoding='utf-8'); t=ast.parse(s)\nbad={'open','eval','exec','compile','__import__','getattr','setattr','delattr','globals','locals','vars','input','breakpoint','help','exit','quit'}\nfor n in ast.walk(t):\n if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.Name) and (n.id in bad or n.id.startswith('__')) or isinstance(n,ast.Attribute) and n.attr.startswith('__'): raise RuntimeError('rejected external execution feature')\nexec(compile(t,str(p),'exec'),{'__builtins__':{k:v for k,v in vars(__builtins__).items() if k not in bad and not k.startswith('__')}})";
   const test=run('python.exe',['-I','-S','-B','-c',guard,path.join(outdir,task.id+'.py')],3000);
   save(task.id+'-run.json',JSON.stringify(test,null,2));
   if(test.status!==0)throw Error('syntax, safety, unit assertion failure or timeout');r.pass=true;
  }
 }catch(e){r.failure=String(e);}
 results.push(r);
}
const summary={identity:prepared.identity,representation:rep,synthetic_oracle_self_test:selfTest,tasks:results,passed:results.filter(r=>r.pass).length,total:results.length,
 categories:Object.fromEntries([...new Set(results.map(r=>r.category))].map(c=>[c,{passed:results.filter(r=>r.category===c&&r.pass).length,total:results.filter(r=>r.category===c).length}])),
 limitations:'Fourteen custom tasks; passing functional tests does not prove complexity, general capability, or retention against unmeasured original BF16 generation. Errors and truncated outputs remain included.'};
save('summary.json',JSON.stringify(summary,null,2)+'\n');console.log(JSON.stringify(summary,null,2));
```
