from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed

from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from layers.Embed import PatchEmbedding, TokenEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # 加载 LLAMA 模型
        if configs.llm_model == 'LLAMA':
            self.llama_config = GPT2Config.from_pretrained('/mnt/petrelfs/chengdawei/.cache/modelscope/hub/iiBcai/gpt2/')
            #self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm_model = GPT2Model.from_pretrained(
                "/mnt/petrelfs/chengdawei/.cache/modelscope/hub/iiBcai/gpt2/",
                trust_remote_code=True,
                local_files_only=True,
            )
            vocab_path = "/mnt/petrelfs/chengdawei/.cache/modelscope/hub/iiBcai/gpt2/vocab.json"
            merges_path = "/mnt/petrelfs/chengdawei/.cache/modelscope/hub/iiBcai/gpt2/merges.txt"
            self.tokenizer = GPT2Tokenizer.from_pretrained( "/mnt/petrelfs/chengdawei/.cache/modelscope/hub/iiBcai/gpt2/",
                trust_remote_code=True,
                local_files_only=True,
            )

        # 处理 pad_token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # 冻结模型参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content

        self.dropout = nn.Dropout(configs.dropout)

        # Patch Embedding 和 Reprogramming Layer
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer1 = ReprogrammingLayer(
            configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.reprogramming_layer2 = ReprogrammingLayer(
            configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # Label Embedding 和 Output Projection
        self.label_embedding = TokenEmbedding(c_in=1, d_model=configs.d_model)
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(
            configs.enc_in, self.head_nf, self.pred_len, head_dropout=configs.dropout)

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, y, test_flag=False):
        dec_out = self.forecast(x_enc, y, test_flag)
        return dec_out[:, -self.pred_len:, :]

    def forecast(self, x_enc, y, test_flag):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()

        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # 生成 Prompt 并处理嵌入
        promp = self.generate_prompt(x_enc)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # 获取模型设备
        model_device = next(self.llm_model.parameters()).device

        prompt = self.tokenizer(
            text=promp,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).input_ids.to(model_device)  # 将 prompt 移动到模型设备

        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt)  # (batch, prompt_token, dim)

        # 在需要访问参数的地方，使用 GatheredParameters 上下文管理器
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0).to(model_device)


        # 确保在上下文管理器之外，不再引用 word_embeddings

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer1(enc_out, source_embeddings, source_embeddings)
        #y = y.reshape(B * N, self.pred_len, 1).to(x_enc.device)
        #label_embeddings = self.label_embedding(y)
        #label_embed = self.reprogramming_layer2(label_embeddings, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        #llama_enc_out = enc_out
        '''
        if test_flag == False:
            y = y.reshape(B * N, self.pred_len, 1).to(x_enc.device)
            label_embeddings = self.label_embedding(y)
            label_embed = self.reprogramming_layer2(label_embeddings, source_embeddings, source_embeddings)
            llama_enc_out = torch.cat([prompt_embeddings, enc_out, label_embed], dim=1)
        else:
            llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        '''
        




        # 确保 llama_enc_out 在模型设备上
        if llama_enc_out.device != model_device:
            llama_enc_out = llama_enc_out.to(model_device)

        
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def generate_prompt(self, x_enc):
        # 分布特征
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        # 波形特征
        var_values = torch.var(x_enc, dim=1)
        slopes = self.get_slope(x_enc)
        # 时间特征
        lags = self.calcute_lags(x_enc)

        prompt = []

        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            var_values_str = str(var_values[b].tolist()[0])
            slopes_values_str = str(slopes[b].tolist())
            lags_values_str = str(lags[b].tolist())

            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"variance value {var_values_str},"
                f"slope value {slopes_values_str}, "
                f"top 5 lags are : {lags_values_str} <|end_prompt|>"
            )
            prompt.append(prompt_)

        return prompt

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def get_slope(self, x_enc):
        t = torch.arange(x_enc.shape[1]).float().unsqueeze(1).to(x_enc.device)
        slopes = []
        for i in range(x_enc.size(0)):
            y = x_enc[i, :, 0]
            t_mean = torch.mean(t)
            y_mean = torch.mean(y)
            slope = torch.sum((t - t_mean) * (y - y_mean)) / torch.sum((t - t_mean) ** 2)

            slopes.append(slope)
        slopes = torch.stack(slopes)
        return slopes


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        # 确保所有张量在同一设备上
        device = target_embedding.device
        source_embedding = source_embedding.to(device)
        value_embedding = value_embedding.to(device)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
