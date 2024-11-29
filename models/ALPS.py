from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
import math

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from layers.Embed import PatchEmbedding, TokenEmbedding
import transformers
from layers.StandardNorm import Normalize
from utils.SBD import calcSBD

transformers.logging.set_verbosity_error()

#Flattening layer followed by a Linear transformation for the output
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

        # Load the Llama model
        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('/mnt/petrelfs/chengdawei/Boris/llama-7b/7b/')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm_model = LlamaModel.from_pretrained(
                "/Boris/llama-7b/7b/",
                trust_remote_code=True,
                local_files_only=True,
            )

            self.tokenizer = LlamaTokenizer.from_pretrained(
                "/Boris/llama-7b/7b/",
                trust_remote_code=True,
                local_files_only=True
            )

        # Set padding token for Llama tokenizer
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # Freeze model parameters (no training)
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

        self.cross_attention_layer1 = Cross_Attention(
            configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.cross_attention_layer2 = Cross_Attention(
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
        # Normalize the input sequence
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()

        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # Generate the prompt and process the embedding
        promp = self.generate_prompt(x_enc)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # Get model device
        model_device = next(self.llm_model.parameters()).device

        prompt = self.tokenizer(
            text=promp,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).input_ids.to(model_device)  # Move prompt to model device

        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt)


        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0).to(model_device)


        # Patch embedding process
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.cross_attention_layer1(enc_out, source_embeddings, source_embeddings)
        
        if test_flag == False:
            y = y.reshape(B * N, self.pred_len, 1).to(x_enc.device)
            label_embeddings = self.label_embedding(y)
            label_embed = self.cross_attention_layer2(label_embeddings, source_embeddings, source_embeddings)
            llama_enc_out = torch.cat([prompt_embeddings, enc_out, label_embed], dim=1)
        else:
            llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        
        # Pass through the Llama model
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        # Project the output
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # Denormalize the output
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def generate_prompt(self, x_enc):
        # workload value features
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values

        # workload waveform features
        var_values = torch.var(x_enc, dim=1)
        slopes = self.get_slope(x_enc)

        # workload temporal features
        lags = self.calcute_lags(x_enc)
        
        # workload spatial features
        B, T, N = x_enc.size()  # Batch size, Time steps, Number of containers
        x_enc_reshaped = x_enc.permute(0, 2, 1)
        top_5_similar = []

        # Compute the SBD (Similarity-Based Distance) for each pair of containers
        for i in range(N):
            distances = [
                calcSBD(x_enc_reshaped[:, i, :], x_enc_reshaped[:, j, :])  # Calculate the distance between container i and container j
                for j in range(N)
            ]
            distances = torch.stack(distances, dim=-1)  # Stack distances for all containers
            _, top_indices = torch.topk(-distances, self.top_k, dim=-1)  # Get indices of the top 5 smallest distances (most similar)
            top_5_similar.append(top_indices)

        top_5_similar = torch.stack(top_5_similar, dim=1)  # Stack top 5 similar containers for all containers, shape: [B, N, top_k]

        prompt = []  # List to store the generated prompts for each sample

        # Generate the prompt string for each batch sample
        for b in range(x_enc.shape[0]):
            # Convert feature values to strings for inclusion in the prompt
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            var_values_str = str(var_values[b].tolist()[0])
            slopes_values_str = str(slopes[b].tolist())
            lags_values_str = str(lags[b].tolist())
            top_similar_str = ", ".join(map(str, top_5_similar[b].tolist()))  # Convert top 5 similar containers to a string

            # Format the prompt string with all the extracted features
            prompt_ = (
                f"<|start_prompt|>You are a helpful assistant for workload prediction."
                f"Dataset description: {self.description};"  # Dataset description
                f"Task description: Please use theprevious {str(self.seq_len)} steps workload information tp predict the next {str(self.pred_len)} steps ; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"variance value {var_values_str}, "
                f"slope value {slopes_values_str}, "
                f"top 5 lags are: {lags_values_str}, "
                f"top 5 similar containers are: {top_similar_str} <|end_prompt|>"
            )
            prompt.append(prompt_)  # Add the prompt for this batch sample to the list

        return prompt  # Return the list of generated prompts

    
    def calcute_lags(self, x_enc):
        # Perform FFT (Fast Fourier Transform) on the input sequence (x_enc) after permuting its dimensions
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        
        # Calculate cross-correlation by multiplying the FFT results
        res = q_fft * torch.conj(k_fft)
        
        # Apply inverse FFT to get the correlation result in the time domain
        corr = torch.fft.irfft(res, dim=-1)
        
        # Calculate the mean value of the correlation along the time dimension
        mean_value = torch.mean(corr, dim=1)
        
        # Find the top-k largest correlations (lags) that are most significant
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def get_slope(self, x_enc):
        # Create a time vector (t) for each time step in the input sequence
        t = torch.arange(x_enc.shape[1]).float().unsqueeze(1).to(x_enc.device)
        
        slopes = [] 
        
        for i in range(x_enc.size(0)):

            y = x_enc[i, :, 0]
            
            # Compute the mean of the time vector and the input sequence
            t_mean = torch.mean(t)
            y_mean = torch.mean(y)
            
            # Calculate the slope (rate of change) of the sequence using the least squares method
            slope = torch.sum((t - t_mean) * (y - y_mean)) / torch.sum((t - t_mean) ** 2)

            slopes.append(slope)
        
        slopes = torch.stack(slopes)
        return slopes



class Cross_Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(Cross_Attention, self).__init__()

        # Set default value for d_keys if not provided
        d_keys = d_keys or (d_model // n_heads)

        # Define projection layers for query, key, and value embeddings
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        
        # Set the number of attention heads
        self.n_heads = n_heads
        
        # Dropout for attention scores
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape  # Batch size, Sequence length, Features
        S, _ = source_embedding.shape  # Source sequence length, Features
        H = self.n_heads  # Number of attention heads

        # Apply linear projections to the target, source, and value embeddings
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        # Ensure all embeddings are on the same device
        device = target_embedding.device
        source_embedding = source_embedding.to(device)
        value_embedding = value_embedding.to(device)

        # Perform cross-attention
        out = self.cross_attention(target_embedding, source_embedding, value_embedding)

        # Reshape the output for the final projection
        out = out.reshape(B, L, -1)

        # Apply the output projection
        return self.out_projection(out)

    def cross_attention(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape  # Batch size, Sequence length, Attention heads, Embedding size

        # Scale factor for attention scores
        scale = 1. / sqrt(E)

        # Calculate attention scores using the dot product of query and key
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        # Apply softmax to get attention weights and apply dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Calculate the final attention output by applying attention weights to the value embeddings
        final_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return final_embedding

