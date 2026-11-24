"""Bounded A/D/G/H/I smoke on the three verified functional inputs."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7e3e67edbbed1bf9888184d9df282b700a323964"
IMAGE_SHA = "774333197d9554b7063303aff10d52bbec1a952f72e54e5fe3d2a7671bef55aa"
PROMPT = "Is there a chair in the image?\nAnswer with Yes or No only."
CASES = {
    IMAGE_SHA: (576, PROMPT),
    "f781f3ff43240353be20f4e58b67a70db2226ff95a7b15bba4d04db9cf97ed0d":
        (832, "Is there a chair in the image?"),
    "09c8f08c56aa4d48f6b4949521e121c64a10e93ac7b2c526e8d47de7b3a92814":
        (1088, "Please examine the whole image carefully before answering the following question: Is there a chair in the image?"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",required=True,type=Path)
    parser.add_argument("--image",required=True,type=Path)
    parser.add_argument("--run-gpu",action="store_true",help="Future authorized run only, at most one generation per method")
    args = parser.parse_args()
    if (args.snapshot.name!=REVISION or not (args.snapshot/"config.json").is_file()
            or not (args.snapshot/"model.safetensors").is_file()):
        raise ValueError("Expected pinned local SmolVLM snapshot")
    image_sha = hashlib.sha256(args.image.read_bytes()).hexdigest()
    if image_sha not in CASES:
        raise ValueError("Smoke requires an allowlisted functional image")
    expected_n, prompt = CASES[image_sha]
    if not args.run_gpu:
        print("Preflight PASS: pinned paths and non-evaluation image. No model/GPU execution.")
        return

    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1")
    sys.path.insert(0,str(ROOT/"src"))
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from coverpruner_attention_probe import native_layout, first_attention_probe
    from coverpruner_checkpoint import atomic, envelope
    from coverpruner_decoder import compress_visual_tokens, greedy_decode
    from coverpruner_coverage import coverage_state
    from coverpruner_execution_profile import validate_evaluation_packet
    from coverpruner_spatial_selection import prepare_problem, select_method
    from study5_layout import processor_case
    from study5_selection import METHODS, select_all
    from study5_scoring import parse_answer, official_answer

    if not torch.cuda.is_available() or "1660 SUPER" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Use the documented GTX 1660 SUPER GPU")
    torch.set_num_threads(1)
    run = ROOT/"evidence"/f"SMOKE_N{expected_n}.json"
    if expected_n!=576:
        prior = json.loads((run.parent/"SMOKE_N576.json").read_bytes())["payload"]
        if prior["status"]!="complete" or [row["selection"]["method"] for row in prior["methods"]]!=list(METHODS):
            raise ValueError("N576 must complete before additional smoke cases")
    with run.open("x",encoding="utf-8") as record:
        record.write('{"status":"started"}\n')
    record=dict(status="running",snapshot=REVISION,image_sha256=image_sha,image=str(args.image),gpu=torch.cuda.get_device_name(0),
                vram_total_bytes=torch.cuda.get_device_properties(0).total_memory,python=sys.version.split()[0],
                torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
                expected_n=expected_n,prompt=prompt,methods=[],reserved=[],tensor_dumps=0)
    def save():
        atomic(run,envelope(record))
    save()
    try:
        processor=AutoProcessor.from_pretrained(str(args.snapshot),local_files_only=True)
        inputs,pos,layout=processor_case(processor,args.image,prompt)
        if layout.n!=expected_n:
            raise ValueError("Live visual count differs from verified case")
        model=AutoModelForImageTextToText.from_pretrained(
            str(args.snapshot),local_files_only=True,dtype=torch.float16,attn_implementation="sdpa"
        ).eval().cuda()
        record["live_layout"]=[layout.n,layout.crop_rows,layout.crop_cols]
        record["model_dtype"]=str(next(model.parameters()).dtype)
        if model.config.image_token_id!=processor.image_token_id or record["model_dtype"]!="torch.float16":
            raise ValueError("Model and Processor image token or dtype mismatch")
        record["model_processor_image_token_id"]=processor.image_token_id
        base=model.model
        text=base.text_model
        if text.config._attn_implementation!="sdpa" or base.vision_model.config._attn_implementation!="sdpa":
            raise ValueError("Study4 SDPA profile drift")
        with torch.inference_mode():
            pixels=inputs["pixel_values"].to("cuda",dtype=torch.float16)
            pmask=inputs["pixel_attention_mask"].cuda()
            encoded=base.get_image_features(pixels,pixel_attention_mask=pmask,output_attentions=False,output_hidden_states=False).pooler_output
            ids=inputs["input_ids"].cuda()
            valid=inputs["attention_mask"].cuda()
            merged=base.inputs_merger(ids,base.get_input_embeddings()(ids),encoded)
            projected=encoded.reshape(layout.n,-1)
            if not torch.equal(merged[0,list(pos.visual)],projected):
                raise ValueError("Visual/Projector ordering mismatch")
            cfg=copy.deepcopy(text.config)
            cfg._attn_implementation="eager"
            positions,mask=native_layout(cfg,merged,valid)
            probe=first_attention_probe(text.layers[0],text.rotary_emb,merged,valid,positions,mask,pos.instruction,pos.visual)
            if set(probe.attention.allowed_visual_counts)!={layout.n}:
                raise ValueError("Probe cannot address the complete visual pool")
            problem=prepare_problem(projected.float().cpu().numpy(),probe.relevance,layout=layout,sequence_indices=pos.visual)
            record["query_score_sha256"]=hashlib.sha256(problem.query_relevance.tobytes()).hexdigest()
            all_methods=select_all(problem)
            rank=tuple(sorted(range(layout.n),key=lambda j:(-float(problem.query_relevance[j]),j)))
            a,d,g,h,i=(all_methods[arm] for arm in METHODS)
            if (h.selection_order!=rank[:64] or g.seed_indices!=rank[:25]
                    or i.seed_indices!=d.seed_indices
                    or i.selection_order[25:]!=tuple(j for j in rank if j not in set(d.seed_indices))[:39]):
                raise ValueError("Query rank or shared seed invariant failed")
            for arm in ("A","D"):
                actual=all_methods[arm]
                baseline=select_method(arm,problem).greedy
                if (actual.selection_order!=baseline.selection_order or actual.seed_indices!=baseline.seed_indices
                        or actual.sorted_sequence_indices!=baseline.sorted_sequence_indices
                        or actual.greedy_objectives!=baseline.objective_values):
                    raise ValueError(f"Study4 {arm} live input parity failed")
            record["invariants"]=dict(shared_query_score=True,study4_live_parity=["A","D"],
                h_global_top64=True,g_global_top25=True,i_uses_d_seed=True,i_query_completion=True,
                query_score_finite=bool(np.isfinite(problem.query_relevance).all()))
            weights=problem.weights.astype(np.float64)
            references={}
            for arm,selection in all_methods.items():
                n=len(selection.selection_order)
                if n!=64 or len(set(selection.selection_order))!=64 or tuple(sorted(selection.sorted_sequence_indices))!=selection.sorted_sequence_indices:
                    raise ValueError(f"Invalid K64 or order in {arm}")
                initial=float(coverage_state(problem.utilities,selection.seed_indices) @ weights)
                final=float(coverage_state(problem.utilities,selection.selection_order) @ weights)
                if not np.isclose(initial,selection.initial_coverage,atol=.001,rtol=1e-6) or not np.isclose(final,selection.final_coverage,atol=.001,rtol=1e-6):
                    raise ValueError(f"Coverage mismatch in {arm}")
                if arm in ("D","G") and (len(selection.greedy_objectives)!=40 or len(selection.seed_indices)!=25 or selection.greedy_objectives[0]!=selection.initial_coverage):
                    raise ValueError(f"Seed25 coverage not carried in {arm}")
                references[arm]={"initial_fp64":initial,"final_fp64":final}
            record["coverage_reference"]=references
            record["shared_peak_bytes"]=torch.cuda.max_memory_allocated()
            save()
            eos=model.generation_config.eos_token_id
            eos_ids=tuple(eos) if isinstance(eos,list) else (() if eos is None else (eos,))
            for arm in METHODS:
                selection=all_methods[arm]
                packet=compress_visual_tokens(ids,merged,valid,pos.visual,selection.sorted_visual_indices,pos.instruction,
                                              image_token_id=processor.image_token_id)
                profile=validate_evaluation_packet(packet,text.config._attn_implementation)
                record["reserved"].append(arm)
                save()
                torch.cuda.reset_peak_memory_stats()
                result=greedy_decode(text,model.lm_head,packet,max_new_tokens=10,eos_token_ids=eos_ids)
                if not torch.isfinite(result.prediction_logits).all():
                    raise ValueError(f"Nonfinite logits in {arm}")
                if packet.selected_original_positions!=selection.sorted_sequence_indices:
                    raise ValueError(f"Decoder position parity failed in {arm}")
                raw=processor.tokenizer.decode(result.token_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                record["methods"].append(dict(selection=asdict(selection),profile=profile,token_ids=result.token_ids,
                     response=raw,strict=parse_answer(raw),official_compatible=official_answer(raw),
                     cache_trace=result.session.trace,prefill_seconds=result.session.prefill_seconds,
                     decode_seconds=result.session.decode_seconds,finite_logits=True,peak_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                print(f"N{expected_n} {arm} complete",flush=True)
                save()
                del result
            record["status"]="complete"
            save()
    except (Exception,KeyboardInterrupt) as exc:
        record["status"]="failed"
        record["error"]=str(exc)
        save()
        raise


if __name__=="__main__":
    main()
