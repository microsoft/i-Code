from .header import *
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

import torch
from torch.nn.utils import rnn

from einops import rearrange
import torchvision.transforms as T

from ...cfg_helper import model_cfg_bank
from ..common.get_model import get_model
import torchvision.transforms as tvtrans

from einops import rearrange, repeat
from core.datasets.instruct_datasets.utils import FEATURE_ID

import sys
sys.path.append('core/models/mmllama/ImageBind')
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

def flatten(l):
    return [item for sublist in l for item in sublist]

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False
        

def gather_true_values(array_0, array_1_bool):
    # Check that the input shapes are compatible
    if array_0.shape[:2] != array_1_bool.shape:
        raise ValueError("Shapes of arrays are not compatible.")
    
    # Collect the slices based on the mask for each 'a'
    slices = [array_0[i][array_1_bool[i]] for i in range(array_0.shape[0])]
    
    # Stack the slices together to get the desired shape
    result = torch.stack(slices, dim=0)
    
    return result
    
def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0: # the first human turn
            assert role == 'human'
            text = '</Perception> ' + turn['value']
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = turn['value']
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                text = turn['value']
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()

PROMPT_START = '<Perception>'
class OpenLLAMAPEFTModel(nn.Module):

    '''LoRA for LLaMa model'''

    def __init__(self, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args
        codi_ckpt_path = args['codi_ckpt_path']
        vicuna_ckpt_path = args['vicuna_ckpt_path']
        max_tgt_len = args['max_tgt_len']
        stage = args['stage']
        self.device = args['device']
        self.dtype = args['dtype']

        self.return_full_seq = args['clip_return_full_seq']
        self.perception_len_image = args['perception_len']
        self.perception_len_audio = args['perception_len']

        print (f'Initializing encoder from {codi_ckpt_path} ...')
        self.imagebind = imagebind_model.imagebind_huge(pretrained=True)

        from omegaconf import OmegaConf
        import sys
        sys.path.append('core/models/mmllama')
        from ldm.util import instantiate_from_config
        config = OmegaConf.load(f"configs/stable-diffusion/v2-1-stable-unclip-h-bind-inference.yaml")
        def load_model_from_config(config, ckpt, verbose=False):
            print(f"Loading model from {ckpt}")
            pl_sd = torch.load(ckpt, map_location="cpu")
            if "global_step" in pl_sd:
                print(f"Global Step: {pl_sd['global_step']}")
            sd = pl_sd["state_dict"]
            model = instantiate_from_config(config.model)
            m, u = model.load_state_dict(sd, strict=False)
            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)
            model.eval()
            return model
        self.image_diffuser = load_model_from_config(config, os.path.join(args['data_dir'], "sd2-1/sd21-unclip-h.ckpt"))
        
        print (f'Initializing language decoder from {vicuna_ckpt_path} ...')
        # add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=self.args['lora_r'], 
            lora_alpha=self.args['lora_alpha'], 
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path, cache_dir=args['data_dir'], use_auth_token='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', torch_dtype=torch.float16, use_flash_attention_2=True)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_ckpt_path, use_fast=False, cache_dir=args['data_dir'], use_auth_token='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')
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print ('Language decoder initialized.')

        self.perception_type = args['perception_type']
        self.max_tgt_len = max_tgt_len

        if 'clip' in self.perception_type:
            dim_in = 1024
            self.dim_in = dim_in
            dropout = 0.1
            dim_out = self.llama_model.config.hidden_size
            
            self.codi_llama_proj = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.GELU(),
                nn.LayerNorm(dim_out),
                nn.Linear(dim_out, dim_out),
            )
            
            self.codi_llama_decode = nn.Sequential(
                nn.Linear(dim_out, dim_in),
                nn.GELU(),
                nn.LayerNorm(dim_in),
                nn.Linear(dim_in, dim_in),
            )

    def prompt_wrap(self, encoder_embeds_raw, input_ids, target_ids, attention_mask):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        encoder_embeds_all = []
        for encoder_embeds in encoder_embeds_raw:
            if encoder_embeds is not None:
                embeds = self.codi_llama_proj(encoder_embeds)
                if embeds.ndim == 3:
                    embeds = embeds.flatten(0,1)
                encoder_embeds_all.append(embeds)
            else:
                encoder_embeds_all.append(None)
                
        batch_size = len(input_ids)

        perception_indices = input_ids==FEATURE_ID
        device = input_ids.device
        llm_embeds = self.llama_model.model.model.embed_tokens(input_ids)
        input_embeds = torch.empty(llm_embeds.shape).to(llm_embeds)
        input_embeds[~perception_indices] = llm_embeds[~perception_indices]
        # print(len(encoder_embeds_all), encoder_embeds_all[0].shape, torch.cat(encoder_embeds_all, 0).shape, input_embeds[perception_indices].shape)
        tmp_encoder_embeds_all = [item for item in encoder_embeds_all if item != None]
        if len(tmp_encoder_embeds_all) > 0:
            concated_embeds = torch.cat(tmp_encoder_embeds_all, 0)
            if input_embeds[perception_indices].shape == concated_embeds[:input_embeds[perception_indices].shape[0]].shape:
                input_embeds[perception_indices] = concated_embeds[:input_embeds[perception_indices].shape[0]]
            else:
                print(input_embeds[perception_indices].shape, concated_embeds[:input_embeds[perception_indices].shape[0]].shape)
        
        # create regression targets
        regression_targets_vae = []
        if 'clip' in self.perception_type:
            regression_targets = (
                torch.ones([input_embeds.shape[0], input_embeds.shape[1], self.dim_in]).to(self.device).to(input_embeds.dtype).fill_(-100)  
            ) # bsz x (1 + s1 + 1)
        else:
            regression_indices = (
                torch.ones([input_embeds.shape[0], input_embeds.shape[1]]).to(self.device).to(input_embeds.dtype).fill_(0)  
            ) # bsz x (1 + s1 + 1)
            
        regression_targets_vae = []
        for i in range(batch_size):
            # print(target_ids[i])
            if encoder_embeds_all[i] is not None:
                target_replace = torch.where(target_ids[i]==FEATURE_ID)[0]
                # input_replace = torch.where(input_ids[i]==FEATURE_ID)[0]
                if len(target_replace) > 0:
                    # s_index = len(input_replace)//self.perception_len  - len(target_replace)//self.perception_len
                    targets_tmp = encoder_embeds_raw[i][-1:].squeeze().to(encoder_embeds_all[0])
                    if self.perception_len > 1:
                        targets_tmp = targets_tmp[:-self.perception_len+1]
                    if 'clip' in self.perception_type:
                        try:
                            # print(regression_targets[i][target_replace].shape, targets_tmp.shape)
                            regression_targets[i][target_replace] = targets_tmp
                        except Exception as e:
                            print(e)
                            regression_targets[i][target_replace] = -100
                    else:
                        try:
                            # print(target_replace)
                            regression_indices[i][target_replace] = 1
                            regression_targets_vae.append(targets_tmp)
                        except Exception as e:
                            print(e)
                target_ids[i][target_replace] = -100
        
                
        bos = torch.ones([batch_size, 1],
                         dtype=input_ids.dtype,
                         device=input_embeds.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        input_embeds = torch.cat([bos_embeds, input_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
        atts_prefix = torch.ones([batch_size, 1], dtype=torch.long, device=input_embeds.device) # bsz x (1 + s1 +1)
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)

        if 'clip' in self.perception_type:
            empty_regression_targets = regression_targets[:, :1].fill_(-100)  
            regression_targets = torch.cat([empty_regression_targets, regression_targets], dim=1) # bsz x (1 + s)
        elif self.perception_type == 'vae':
            empty_regression_indices = regression_indices[:, :1].fill_(0)  
            regression_indices = torch.cat([empty_regression_indices, regression_indices], dim=1) # bsz x (1 + s)
            if regression_targets_vae:
                regression_targets_vae = torch.stack(regression_targets_vae, 0)
            regression_targets = (regression_indices, regression_targets_vae)
        
        empty_targets = (
            torch.ones([batch_size, 1], dtype=torch.long).to(self.device).fill_(-100)  
        ) # bsz x (1)
        target_ids = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s)
        return input_embeds, target_ids, attention_mask, regression_targets
    
    def forward(self, inputs):
        if 'image' in inputs:
            self.perception_len = self.perception_len_image
            encoder_embeds_all = []
            for image_list in inputs['image']:
                if 'clip' in self.perception_type:
                    with torch.no_grad():
                        encoder_embeds_unnormed = self.imagebind({ModalityType.VISION: data.load_and_transform_vision_data(image_list, self.device)})['vision']
                        encoder_embeds = encoder_embeds_unnormed / torch.norm(encoder_embeds_unnormed[:, :1], dim=-1, keepdim=True)

                encoder_embeds_all.append(encoder_embeds)
        elif 'text' in inputs:
            self.perception_len = 1
            encoder_embeds_all = []
            for text_list in inputs['text']:
                if len(text_list) > 0:
                    with torch.no_grad():
                        encoder_embeds = self.imagebind({ModalityType.TEXT: data.load_and_transform_text(text_list, self.device)})['text']
                        # encoder_embeds = encoder_embeds_unnormed / torch.norm(encoder_embeds_unnormed[:, :1], dim=-1, keepdim=True)
                        # print(encoder_embeds.shape)
                    encoder_embeds_all.append(encoder_embeds)
                else:
                    encoder_embeds_all.append(None)
        elif 'audio' in inputs:
            self.perception_len = 1
            encoder_embeds_all = []
            for audio_list in inputs['audio']:
                if 'clip' in self.perception_type:
                    with torch.no_grad():
                        encoder_embeds = self.imagebind({ModalityType.AUDIO: data.load_and_transform_audio_data(audio_list, self.device)})['audio']
                        # encoder_embeds = encoder_embeds_unnormed / torch.norm(encoder_embeds_unnormed[:, :1], dim=-1, keepdim=True)

                encoder_embeds_all.append(encoder_embeds)
                
        target_ids = inputs['output_ids']
        attention_mask = inputs['attention_mask']
        input_ids = inputs['input_ids']

        inputs_embeds, targets, attention_mask, regression_targets = self.prompt_wrap(encoder_embeds_all, input_ids, target_ids, attention_mask)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )
        loss = outputs.loss
        
        # calculate regression loos
        if 'clip' in self.perception_type:
            indice_mask = (regression_targets[:, 1:, -1] != -100)
            if indice_mask.sum() > 0:
                regression_logits = self.codi_llama_decode(outputs.hidden_states[-1][:, :-1])
                regression_logits_normed = regression_logits/torch.norm(regression_logits, dim=-1, keepdim=True)
                regression_labels = regression_targets[:, 1:]

                # Reshape embeddings
                b, n, c = regression_logits_normed.shape
                cosine_sim = F.cosine_similarity(regression_labels.transpose(1,2), regression_logits_normed.transpose(1,2))
                feature_loss = -cosine_sim
                feature_loss = feature_loss[indice_mask].mean() * 0.1
            else:
                feature_loss = torch.tensor(0.0)
                
            if self.perception_type == 'clip-diffusser-back-proop' and 'image' in inputs:
                if indice_mask.sum() > 0:
                    self.image_diffuser = self.image_diffuser.float()
                    try:
                        num_trues = indice_mask.sum(dim=1)
                        assert num_trues.min() == num_trues.max()
                        b, _, c = regression_logits.shape
                        # Extract values
                        diffusion_inputs = regression_logits[indice_mask.unsqueeze(-1).expand_as(regression_logits)].view(b, num_trues[0], c)
                        diffusion_inputs = diffusion_inputs[:, :1]
                    except Exception as e:
                        print(e)
                        diffusion_inputs = torch.zeros_like(regression_logits[:, :1]).to(next(self.codi.parameters()).dtype)

                    c_adm = diffusion_inputs.squeeze(1).float()
                    batch_size = c_adm.shape[0]
                    if self.image_diffuser.noise_augmentor is not None:
                        c_adm, noise_level_emb = self.image_diffuser.noise_augmentor(c_adm, noise_level=(
                                torch.ones(batch_size) * self.image_diffuser.noise_augmentor.max_noise_level * 0.5).long().to(c_adm.device))
                        # assume this gives embeddings of noise levels
                        c_adm_noise = torch.cat((c_adm, noise_level_emb), 1).float()
                    uc_encoder = self.image_diffuser.get_learned_conditioning(batch_size * ['longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality']).float()
                    c = {"c_crossattn": [uc_encoder], "c_adm": c_adm_noise}
                    # print(uc_encoder.dtype, c_adm_noise.dtype, )
                    x_inputs = []
                    BICUBIC = T.InterpolationMode.BICUBIC
                    
                    for image_list in inputs['image']:
                        tmp_image = image_list[-1]
                        transforms = T.Compose([
                            T.RandomCrop(min(tmp_image.size)),
                            T.Resize(
                                (768, 768),
                                interpolation=BICUBIC,
                            ),
                            T.ToTensor(),
                        ])
                        x_inputs.append(transforms(tmp_image))
                    
                    x_inputs = torch.stack(x_inputs).to(self.device).float()
                    latents = self.image_diffuser.encode_first_stage(x_inputs)
                    latents = self.image_diffuser.get_first_stage_encoding(latents)
                    # print(latents.device, c["c_crossattn"][0].device, c["c_adm"].device)
                    diffusion_loss = self.image_diffuser(latents, c)[0]
                    
                    feature_loss += diffusion_loss
                    # print(feature_loss)
                else:
                    feature_loss += torch.tensor(0.0)
        elif self.perception_type == 'vae':
            regression_indices, regression_targets_vae = regression_targets
            if len(regression_targets_vae) > 0:
                hidden_states = outputs.hidden_states[-1][:, :-1]
                indice_mask = regression_indices.bool()[:, 1:]
                regression_logits = self.codi_llama_decode(gather_true_values(hidden_states, indice_mask))
                feature_loss = torch.square(regression_targets_vae-regression_logits).mean()
            else:
                feature_loss = torch.tensor(0.0)
        # calculate the token accuarcy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, :-1]    # [B, S-1]
        labels = targets[:, 1:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask    # [B*S]
        if valid_mask.sum().item() > 0:
            gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        else:
            gen_acc = torch.tensor(0.0)
        return loss, gen_acc, feature_loss

    def extract_multimodal_feature(self, inputs):
        features = {}
        if inputs['text']:
            text_embeds = [self.codi.clip_encode_text([item])[0] for item in inputs['text']]
            features['#text#'] = text_embeds
        if inputs['image']:
            if 'clip' in self.perception_type:
                with torch.no_grad():
                    image_embeds = self.imagebind({ModalityType.VISION: data.load_and_transform_vision_data(inputs['image'], self.device).to(self.dtype)})['vision']
                    image_embeds = image_embeds / torch.norm(image_embeds[:, :1], dim=-1, keepdim=True)
            elif self.perception_type == 'vae':
                image_embeds = [self.codi.autokl_encode(item)[0] for item in inputs['image']]
            
            features['#image#'] = image_embeds
        if inputs['audio']:
            audio_embeds, _ = [self.codi.clap_encode_audio(item)[0] for item in inputs['audio']]
            features['#audio#'] = audio_embeds
        if inputs['video']:
            video_embeds, _ = [self.codi.clip_encode_vision(item)[0] for item in inputs['video']]
            features['#video#'] = video_embeds
        return features
    
    def prepare_generation_embedding(self, inputs):
        prompt = inputs['prompt']
        if len(inputs['modality_embeds']) == 1:
            feature_embeds = inputs['modality_embeds'][0]
        else:
            feature_embeds = self.extract_multimodal_feature(inputs)
            inputs['modality_embeds'].append(feature_embeds)

        return self.process_and_concat_embeds(prompt, feature_embeds)
    
    def process_and_concat_embeds(self, prompt, feature_embeds):
        batch_size = len(prompt)
        context_embeds = []
        for text in prompt[0].split(' '):
            if any(tag in text for tag in ['#image#', '#audio#', '#video#', '#text#']):
                modality_tag, index = text.split('_')
                projected_embed = self.codi_llama_proj(feature_embeds[modality_tag][int(index)].unsqueeze(0))
                context_embeds.append(projected_embed)
            else:
                tokens = self.llama_tokenizer(text, 
                return_tensors="pt", add_special_tokens=False).to(self.device)
                embeds = self.llama_model.model.model.embed_tokens(tokens.input_ids) # bsz x s1 x embed_dim
                context_embeds.append(embeds)
        
        bos = torch.ones([batch_size, 1],
                         dtype=tokens.input_ids.dtype,
                         device=tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        inputs_embeds = torch.cat([bos_embeds] + context_embeds, dim=1) # bsz x (1+s1+1+s2) x embed_dim
        return inputs_embeds

    def generate(self, inputs):
        '''
            inputs = {
                'text': optional,
                'image': optional,
                'audio': optional
                'video': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature
                'modality_embeds': None or torch.tensor
                'modality_cache': save the image cache
            }
        '''
        input_embeds = self.prepare_generation_embedding(inputs)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        output_ids = outputs.sequences[0][1:]
        output_text = self.llama_tokenizer.decode(output_ids, skip_special_tokens=True)
        hidden_states_last = [] 
        for item in outputs.hidden_states:
            hidden_states_last.append(item[-1].squeeze())
        
        def find_pattern(sequence1, sequence2, pattern):
            results = []
            pattern_length = len(pattern)
            for i in range(len(sequence1) - pattern_length + 1):
                if sequence1[i:i+pattern_length].detach().cpu().tolist() == pattern:
                    j = i + pattern_length - 1
                    if j + 1 < len(sequence2):
                        results.append(sequence2[j + 1])
            return results

        hidden_states_filtered = find_pattern(output_ids, hidden_states_last, [529, 29886, 29958])
        if hidden_states_filtered:
            hidden_states_filtered = torch.stack(hidden_states_filtered)
            output_features = self.codi_llama_decode(hidden_states_filtered)
        else:
            output_features = None
        return output_text, output_features

    def decode(self, z, xtype):
        net = self.codi
        z = z.to(self.device)
        if xtype == 'image':
            x = net.autokl_decode(z)
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
            x = [tvtrans.ToPILImage()(xi) for xi in x]
            return x
        
        elif xtype == 'video':
            num_frames = z.shape[2]
            z = rearrange(z, 'b c f h w -> (b f) c h w')
            x = net.autokl_decode(z) 
            x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
            
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
            video_list = []
            for video in x:
                video_list.append([tvtrans.ToPILImage()(xi) for xi in video])
            return video_list

        elif xtype == 'text':
            prompt_temperature = 1.0
            prompt_merge_same_adj_word = True
            x = net.optimus_decode(z, temperature=prompt_temperature)
            if prompt_merge_same_adj_word:
                xnew = []
                for xi in x:
                    xi_split = xi.split()
                    xinew = []
                    for idxi, wi in enumerate(xi_split):
                        if idxi!=0 and wi==xi_split[idxi-1]:
                            continue
                        xinew.append(wi)
                    xnew.append(remove_duplicate_word(' '.join(xinew)))
                x = xnew
            return x
        
        elif xtype == 'audio':
            x = net.audioldm_decode(z)
            x = net.mel_spectrogram_to_waveform(x)
            return x
        
    def inference(self, inputs):
        output_text, output_features = self.get_llm_outputs(inputs)
        output_image = self.generate_image(output_features)
        return output_text, output_image

    def generate_image(self, features, xtype, image_size=768, n_samples=1, ddim_steps=50, scale=7.5):
        from ldm.models.diffusion.ddim import DDIMSampler
        sampler = DDIMSampler(self.image_diffuser, device=self.device)

        c_adm = repeat(features, '1 ... -> b ...', b=n_samples)[:, 0]
        if self.image_diffuser.noise_augmentor is not None:
            c_adm, noise_level_emb = self.image_diffuser.noise_augmentor(c_adm, noise_level=(
                    torch.ones(n_samples) * self.image_diffuser.noise_augmentor.max_noise_level * 0.5).long().to(c_adm.device))
            # assume this gives embeddings of noise levels
            c_adm_noise = torch.cat((c_adm, noise_level_emb), 1)
        uc_encoder = self.image_diffuser.get_learned_conditioning(n_samples * ['longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'])
        c_adm_noise = c_adm_noise.to(uc_encoder)
        print(c_adm_noise.dtype, uc_encoder.dtype)
        uc = {"c_crossattn": [uc_encoder], "c_adm": torch.zeros_like(c_adm_noise)}
        c = {"c_crossattn": [uc_encoder], "c_adm": c_adm_noise}
        
        samples, _ = sampler.sample(S=ddim_steps,
                                    conditioning=c,
                                    batch_size=n_samples,
                                    shape=[4, image_size//8, image_size//8],
                                    verbose=False,
                                    unconditional_guidance_scale=scale,
                                    unconditional_conditioning=uc,
                                    eta=0.0,
                                    x_T=None)
        
        x_samples = self.image_diffuser.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

        images = []
        from PIL import Image
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            images.append(img)
        return images