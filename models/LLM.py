from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding, TokenEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)#展平层，从倒数第二维开始展平
        self.linear = nn.Linear(nf, target_window)#线性层
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
        self.top_k = 5 #
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len #用于处理时间序列数据的分块操作，代表一个块的大小
        self.stride = configs.stride #用于定义数据块之间的步长或间隔。用于滑动窗口操作时确定每次滑动的距离。

        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('/home/user/Boris/Time-LLM-main/llama-7b/7b/')
            #self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                     "/home/user/Boris/Time-LLM-main/llama-7b/7b/",
                    #'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                     "/home/user/Boris/Time-LLM-main/llama-7b/7b/",
                    #'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                     "/home/user/Boris/Time-LLM-main/llama-7b/7b/tokenizer.model",
                    #'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(#tokenizer是用于处理和预处理文本数据的工具
                     "/home/user/Boris/Time-LLM-main/llama-7b/7b/tokenizer.model",
                    #'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )

        else:
            raise Exception('LLM model is not defined')
        
        #处理变长序列
        if self.tokenizer.eos_token:#eos_token是end of sentence token
            self.tokenizer.pad_token = self.tokenizer.eos_token#pad token是填充标记
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False#冻结模型参数

        if configs.prompt_domain:#决定是否使用用户提供的自定义描述内容（content），还是使用默认的描述内容
            self.description = configs.content

        self.dropout = nn.Dropout(configs.dropout)

        #patch embedding将输入数据分割成固定大小的补丁（patch），然后将每个补丁转换为嵌入向量
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)#patch_len是每个补丁包含的数据点数；stride是相邻补丁间的距离

        self.word_embeddings = self.llm_model.get_input_embeddings().weight #从预训练模型中获取所有词汇的嵌入向量。
        self.vocab_size = self.word_embeddings.shape[0] #词汇表的大小
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)#把词汇嵌入向量维度从vocab_size映射到1000维度
        #self.mapping_layer2 = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer1 = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.reprogramming_layer2 = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        #self.cross_attention = CrossAttention(configs.n_heads)
        self.label_embedding = TokenEmbedding(c_in = 1, d_model = configs.d_model)
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums 
        #head_nf表示经过前馈网络处理后，所有patch的总维度
        #d_ff表示前馈网络隐层神经元数量

        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, y):
        dec_out = self.forecast(x_enc, y)
        return dec_out[:, -self.pred_len:, :]
        
    def get_top_k_related_containers(self, x_enc, k=5):
        B, T, N = x_enc.size()  # x_enc shape: [batch_size, seq_len, num_containers]
        top_k_indices = []

        # 计算每个容器的相关性
        for i in range(N):
            # 当前容器的数据
            container_i = x_enc[:, :, i]  # shape: [batch_size, seq_len]

            # 计算当前容器与其他容器的相关性（皮尔逊相关系数）
            correlations = []
            for j in range(N):
                container_j = x_enc[:, :, j]  # shape: [batch_size, seq_len]
                
                # 计算每个批次下两个容器的相关性
                corr = torch.zeros(B)  # 用于保存批次内的相关性
                for b in range(B):
                    # 使用皮尔逊相关系数
                    corr[b] = torch.corrcoef(torch.stack((container_i[b], container_j[b])))[0, 1]
                
                correlations.append(corr.mean())  # 保存批次平均相关性
            
            # 获取当前容器最相关的 top k 容器索引（排除自己）
            correlations = torch.stack(correlations)
            #print(correlations.shape)
            top_k = torch.topk(correlations, k=k + 1).indices  # 包括自身，需要+1
            top_k = top_k[top_k != i][:k]  # 排除自己，取前k个

            top_k_indices.append(top_k.tolist())
        
        return top_k_indices



    def forecast(self, x_enc, y):
        #x_enc=x_enc.squeeze(-1)
        x_enc = self.normalize_layers(x_enc, 'norm')
        
        B, T, N = x_enc.size()#[2,20,100]
        
        top_k_indices = self.get_top_k_related_containers(x_enc)
        
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)#这样做是为了将每个特征的数据独立出来，以便于后续的模型处理和计算
        
        #分布特征
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        
        # 下分位数（25th percentile）
        #lower_quartile = torch.quantile(x_enc, 0.25, dim=1)
        #上分位数（75th percentile）
        #upper_quartile = torch.quantile(x_enc, 0.75, dim=1)

        #波形特征
        var_values = torch.var(x_enc, dim=1)
        slopes = self.get_slope(x_enc)    
            
        #时间特征
        lags = self.calcute_lags(x_enc)
        
        #trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            #lower_quartile_str = str(lower_quartile[b].tolist()[0])
            #upper_quartile_str = str(upper_quartile[b].tolist()[0])
            var_values_str = str(var_values[b].tolist()[0])
            slopes_values_str = str(slopes[b].tolist())
            lags_values_str = str(lags[b].tolist())
            
            top_k_str = []
            
            for i, indices in enumerate(top_k_indices):
                related_containers = ', '.join(map(str, indices))
                top_k_str.append(f"Container {i+1}: top 5 related containers are {related_containers}")
            
            top_k_description = " ".join(top_k_str)
            
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"variance value {var_values_str},"
                f"slope value {slopes_values_str}, "
                f"top 5 lags are : {lags_values_str},"
                f"Top 5 related containers: {top_k_description} <|end_prompt|>"
            )
            prompt.append(prompt_)
        
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()#再把维度变为[B,T,N]

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        #return_tensor=pt代表指定返回的格式为pytorch张量
        #padding=True代表将输入文本填充到相同长度
        #truncation=True代表将超过最大长度的文本阶段
        #max_length=2048代表最大长度为2048

        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        #正常的word_embedding的维度是(vocab_size,embedding_dim)
        #经过一次permute后变成了(embedding_dim,vocab_size)
        #之后经过mapping layer(输入为embedding_dim，输出是num_tokens)得到形状为(num_tokens,vocab_size)的新嵌入矩阵
        #最后再经过permute操作将维度顺序变为(vocab_size,num_tokens)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        
        enc_out = self.reprogramming_layer1(enc_out, source_embeddings, source_embeddings)
        
        y = y.reshape(B*N, self.pred_len, 1)
        label_embeddings = self.label_embedding(y.to(torch.bfloat16).to(x_enc.device))
        label_embed = self.reprogramming_layer2(label_embeddings, source_embeddings, source_embeddings)
        #llama_enc_out = self.cross_attention(prompt_embeddings, enc_out)
        
        #print(prompt_embeddings.shape, enc_out.shape, label_embeddings.shape)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out, label_embed], dim=1)
        #print(llama_enc_out.shape)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state#表示最后一个隐层的输出状态，形状为batch_size,seq_length,hidden_size
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')#denorm表示反归一化

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)#对输入序列在时间维度上进行快速傅里叶变化
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)#计算了q_fft和k_fft的共轭成绩，相对于在频域中计算自相关函数
        corr = torch.fft.irfft(res, dim=-1)#逆傅里叶变化
        mean_value = torch.mean(corr, dim=1)#获得个时间步的平均自相关值
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)#找到相关性最强的top_k个滞后值，返回它们的索引
        return lags
    def get_slope(self, x_enc):
        # 创建时间步向量，形状为 [T, 1]
        t = torch.arange(x_enc.shape[1]).float().unsqueeze(1).to(x_enc.device)  # [T, 1]
        # 计算斜率
        slopes = []
        for i in range(x_enc.size(0)):  # 遍历每个特征序列
            y = x_enc[i, :, 0]  # [T] 取出每个特征序列的值
            # 使用最小二乘法计算斜率
            t_mean = torch.mean(t)
            y_mean = torch.mean(y)
            slope = torch.sum((t - t_mean) * (y - y_mean)) / torch.sum((t - t_mean) ** 2)
        
            slopes.append(slope)
        slopes = torch.stack(slopes)  # 将所有斜率收集到一个张量中
        return slopes

class ReprogrammingLayer(nn.Module):
    #通过多头注意力机制，将目标嵌入（target_embedding）与源嵌入（source_embedding）之间进行重新编程，
    # 生成加权后的嵌入表示。这种方法可以在不同特征空间之间建立联系，从而提高模型在特定任务中的表现。
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)#每个多头的key向量维度

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

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        #该方法的核心思想是通过注意力机制，将源嵌入（source_embedding）与目标嵌入（target_embedding）的相似性进行量化，
        # 再利用这个相似性对 value_embedding 进行加权，生成一个新的嵌入表示。这种方式可以灵活地将不同的嵌入信息结合起来，从而在特定任务中提高模型的表现。
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)#计算缩放因子

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)#计算注意力分数

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)#重新计算嵌入

        return reprogramming_embedding

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_model=4096, attention_dropout=0.1):
        super(CrossAttention, self).__init__()

        self.d_keys = d_model // n_heads#每个多头的key向量维度

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, a, b):
        # 线性变换得到Q, K, V
        Q = self.query_projection(a)
        K = self.key_projection(b)
        V = self.value_projection(b)

        # 将Q, K, V分为多头
        Q = Q.view(Q.size(0), -1, self.n_heads, self.d_keys).transpose(1, 2)
        K = K.view(K.size(0), -1, self.n_heads, self.d_keys).transpose(1, 2)
        V = V.view(V.size(0), -1, self.n_heads, self.d_keys).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_keys ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 计算加权输出
        out = torch.matmul(attn_weights, V)

        # 拼接多头输出并进行线性变换
        out = out.transpose(1, 2).contiguous().view(a.size(0), -1, self.d_model)
        #out.transpose(1,2)是将张量在第1维和第2维进行互换
        #-1 是自动推导维度大小的占位符，PyTorch 会根据其余维度的大小自动计算这一维度的值。
        out = self.out_projection(out)

        return out