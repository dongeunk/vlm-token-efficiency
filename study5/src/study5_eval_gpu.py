"""GPU adapter matching the P2 no-padding path. Imports are deferred by CLI."""
import copy
from dataclasses import asdict
import hashlib

import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor

from coverpruner_attention_probe import native_layout, first_attention_probe
from coverpruner_decoder import compress_visual_tokens, greedy_decode
from coverpruner_execution_profile import validate_evaluation_packet
from coverpruner_spatial_selection import prepare_problem
from study5_eval import CONFIG, SUFFIX, require
from study5_layout import processor_case
from study5_selection import select_all


class Backend:
    def __init__(self,snapshot):
        require(torch.__version__==CONFIG["torch"] and transformers.__version__==CONFIG["transformers"],"Runtime version drift")
        require(torch.cuda.is_available() and CONFIG["gpu"] in torch.cuda.get_device_name(0),"Wrong GPU or CUDA unavailable")
        torch.set_num_threads(1)
        self.processor=AutoProcessor.from_pretrained(str(snapshot),local_files_only=True)
        self.model=AutoModelForImageTextToText.from_pretrained(
            str(snapshot),local_files_only=True,dtype=torch.float16,attn_implementation="sdpa").eval().cuda()
        self.base=self.model.model
        self.text=self.base.text_model
        require(self.model.config.image_token_id==self.processor.image_token_id and
                next(self.model.parameters()).dtype==torch.float16 and
                self.text.config._attn_implementation==self.base.vision_model.config._attn_implementation=="sdpa",
                "Processor, model or execution profile drift")
        self.probe_config=copy.deepcopy(self.text.config)
        self.probe_config._attn_implementation="eager"
        eos=self.model.generation_config.eos_token_id
        self.eos=tuple(eos) if isinstance(eos,list) else (() if eos is None else (eos,))
        self.key=None

    @torch.inference_mode()
    def prepare(self,job):
        key=job["image_id"],job["question_id"]
        if self.key!=key:
            self.key=None
            self.data=None
            inputs,pos,layout=processor_case(self.processor,job["image_path"],job["text"]+SUFFIX)
            pixels=inputs["pixel_values"].to("cuda",dtype=torch.float16)
            pmask=inputs["pixel_attention_mask"].cuda()
            projected=self.base.get_image_features(pixels,pixel_attention_mask=pmask,
                output_attentions=False,output_hidden_states=False).pooler_output
            ids=inputs["input_ids"].cuda()
            mask=inputs["attention_mask"].cuda()
            merged=self.base.inputs_merger(ids,self.base.get_input_embeddings()(ids),projected)
            features=projected.reshape(layout.n,-1)
            require(torch.equal(merged[0,list(pos.visual)],features),"Visual order mismatch")
            positions,causal=native_layout(self.probe_config,merged,mask)
            probe=first_attention_probe(self.text.layers[0],self.text.rotary_emb,merged,mask,positions,causal,pos.instruction,pos.visual)
            require(set(probe.attention.allowed_visual_counts)=={layout.n},"Incomplete visual attention")
            problem=prepare_problem(features.float().cpu().numpy(),probe.relevance,layout=layout,sequence_indices=pos.visual)
            self.data=(ids,mask,merged,pos,layout,select_all(problem),hashlib.sha256(problem.query_relevance.tobytes()).hexdigest())
            self.key=key
        ids,mask,merged,pos,layout,arms,score_sha=self.data
        sel=arms[job["method"]]
        packet=compress_visual_tokens(ids,merged,mask,pos.visual,sel.sorted_visual_indices,pos.instruction,
                                     image_token_id=self.processor.image_token_id)
        profile=validate_evaluation_packet(packet,self.text.config._attn_implementation)
        require(packet.selected_original_positions==sel.sorted_sequence_indices,"Selection ordering drift")
        return packet,asdict(sel),profile,score_sha,layout.n

    @torch.inference_mode()
    def generate(self,prepared):
        packet,selection,profile,score_sha,n=prepared
        result=greedy_decode(self.text,self.model.lm_head,packet,max_new_tokens=CONFIG["max_new_tokens"],eos_token_ids=self.eos)
        require(bool(torch.isfinite(result.prediction_logits).all()),"Nonfinite logits")
        return dict(response=self.processor.tokenizer.decode(result.token_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                    token_ids=result.token_ids,selection=selection,profile=profile,query_score_sha256=score_sha,
                    visual_count=n,cache_trace=result.session.trace)

