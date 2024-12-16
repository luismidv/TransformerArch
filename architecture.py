import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    #STARTING FROM KEYS QUERIES AND VALUES
    #LINEAR TRANSFORM THOSE 3 ITEMS
    #COMPUTE SCALE DOT PRODUCT ATTENTION
    #CONCATENATE RESULTS
    #LINEAR TRANSFORM RESULTS
    def __init__(self, model_dimension, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert model_dimension % num_heads == 0

        #Dimensions
        self.model_dimension = model_dimension
        self.num_heads = num_heads
        self.d_k = model_dimension // num_heads

        #Linear layers for transformations
        self.W_query = nn.Linear(model_dimension, model_dimension)
        self.W_key = nn.Linear(model_dimension, model_dimension)
        self.W_value = nn.Linear(model_dimension, model_dimension)
        self.W_output = nn.Linear(model_dimension,model_dimension)

    def scale_dotproduct_attention(self,query, key, values, mask = None):
        atention_scores = torch.matmul(query,key.transpose(-2,1)) / math.sqrt(self.d_k)
        if mask is not None:
            atention_scores = atention_scores.masked_fill(mask == 0, -1e9)

        atention_probs = torch.softmax(atention_scores, dim = 1)
        output = torch.matmul(atention_probs, values)
        return output
    
    def split_heads(self,x):
        #RESHAPE TO HAVE NUM_HEAD FOR MULTIHEAD
        batch_size, seq_length,d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads,self.d_k).transpose
    
    def combine_heads(self,x):
        batch_size,_,seq_length, d_k = x.size()
        return x.tranpose(1,2).contiguous().view(batch_size, seq_length, self.model_dimension)
    
    def forward(self,query,key,value,mask = None):
        query = self.split_heads(self.W_query(query))
        key = self.split_heads(self.W_key(key))
        value = self.split_heads(self.W_value(value))

        atention_output = self.scaled_dot_product_attention(query,key,value,mask)

        output = self.W_o(self.combine_heads(atention_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model, d_feedforw):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_feedforw)
        self.fc2 = nn.Linear(d_feedforw, d_model)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

class PositionEncoding(nn.Module):
    def __init__(self,model_dimension, max_seq_length):
        super(PositionEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length,model_dimension)
        position = torch.arange(0,max_seq_length, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dimension, 2).float()) * -(math.log(10000.0))
        pe = [:, 0::2] = torch.sin(position*div_term)
        pe = [:, 1::2] = torch.cos(position*div_term)

        #REGISTERED AS A BUFFER, PART OF THE MODULE BUT WONT BE CONSIDERED AS PARAMETER
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self,x):
        return x + self.pe[:, :x.size(1)]
    

class Encoder(nn.Moudle):
    def __init__(self, model_dimension, num_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.multi_head = MultiHeadAttention(model_dimension, num_heads)
        self.position_ff = PositionWiseFeedForward(model_dimension, d_ff)
        self.norm1 = nn.LayerNorm(model_dimension)
        self.norm2 = nn.LayerNorm(model_dimension)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        atention_output = self.multi_head(x,x,x,mask)
        x = self.norm1(x + self.dropout_layer(atention_output))
        ff_output = self.position_ff(x)
        x = self.norm2(x + self.dropout_layer(ff_output))

class Decoder(nn.Module):
    def __init__(self,model_dimension, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.multi_head = MultiHeadAttention(model_dimension, num_heads)
        self.cross_head = MultiHeadAttention(model_dimension, num_heads)
        self.position_ff = PositionWiseFeedForward(model_dimension, d_ff)
        self.norm1 = nn.LayerNorm(model_dimension)
        self.norm2 = nn.LayerNorm(model_dimension)
        self.norm3 = nn.LayerNorm(model_dimension)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self,x,enc_output,src_mask,tgt_mask):
        atention_output = self.multi_head(x,x,x,tgt_mask)
        x = self.norm1(x + self.dropout_layer(atention_output))
        atention_output = self.cross_head(x,enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout_layer(atention_output))
        ff_output = self.position_ff(x)
        x = self.norm3(x + self.dropout_layer(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, model_dimension, num_heads, num_layers,d_ff,max_seq_length, dropout):
        super(Transformer,self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, model_dimension)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, model_dimension)
        self.positional_encoding = PositionEncoding(model_dimension, max_seq_length)

        self.encoder_layers = nn.ModuleList([Encoder(model_dimension, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([Decoder(model_dimension, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self. fc = nn.Linear(model_dimension, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self,src,tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal = 1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, src, tgt)
        src_mask, tgt_mask = self.generate_mask(src,tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(tgt)))

        





