import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
    

class PepGenGPT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.sep_idx = self.config["alphabet"].get_idx("<sep>")
        self.eos_idx = self.config["alphabet"].get_idx("<eos>")
        self.pad_idx = self.config["alphabet"].get_idx("<pad>")
        
        if self.config['use_esm_token_embedding']:
            self.peptide_embedder = nn.Embedding(
                len(config['alphabet']),
                self.config['esm_dim'],
                padding_idx=self.pad_idx
            )
            self.peptide_embedder.load_state_dict(self.config["esm_token_embedding"])
            
            # load esm embedder
            self.peptide_embedding_map = nn.Linear(
                self.config['esm_dim'], 
                self.config['hidden_dim']
            )
        else:
            self.peptide_embedder = nn.Embedding(
                len(config['alphabet']),
                self.config['hidden_dim'],
                padding_idx=self.pad_idx
            )
        
        # reduce dimensionality of input embedding
        self.esm_embedding_map = nn.Linear(self.config['esm_dim'], self.config['hidden_dim'])

        # positional embeddings as defined in Attention is All You Need
        positions = torch.arange(config["max_seq_length"], dtype=torch.float)
        freqs = torch.pow(10000, -torch.arange(0, torch.div(self.config['hidden_dim'], 2), self.config['hidden_dim']))

        self.full_positional_embedding = torch.zeros(config["max_seq_length"], self.config['hidden_dim'])

        trig_arguments = torch.matmul(positions.unsqueeze(-1), torch.t(freqs.unsqueeze(-1)))
        self.full_positional_embedding[:, torch.arange(0, self.config['hidden_dim'], 2)] = torch.sin(trig_arguments)
        self.full_positional_embedding[:, torch.arange(1, self.config['hidden_dim'], 2)] = torch.cos(trig_arguments)

        self.full_positional_embedding = nn.Parameter(self.full_positional_embedding, requires_grad=config['learnable_positional_embedding'])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config['hidden_dim'],
            nhead=self.config['num_heads'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            layer_norm_eps=self.config['layer_norm_eps'],
            batch_first=True
            # TODO: fix norm_first
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.config['num_layers'])
        self.logit_map = nn.Linear(self.config['hidden_dim'], len(config['alphabet']))

    def positional_embedding(self, size):
        return self.full_positional_embedding[:size, :]

    def embed_peptide(self, peptide_input):
        embedding = self.peptide_embedder(peptide_input)
        if self.config['use_esm_token_embedding']:
            embedding = self.peptide_embedding_map(embedding)
        
        return embedding
    
    def forward(self, peptide_input, raw_partner_embedding, padding_mask, causal_mask):
        peptide_length = peptide_input.shape[1]
        
        peptide_embedding = self.embed_peptide(peptide_input)
        partner_embedding = self.esm_embedding_map(raw_partner_embedding)

        x = torch.cat([partner_embedding, peptide_embedding], dim=1)

        if self.config['positional_embed_peptide_only']:
            x[:, -peptide_length:, :] += self.positional_embedding(peptide_length).unsqueeze(0)
        else:
            x += self.positional_embedding(x.shape[1]).unsqueeze(0)
                
        x = self.encoder(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        
        x = self.logit_map(x)
        
        # we only care about the peptide predictions
        x = x[:, -peptide_length:, :]
        return x
            
    def predict(self, partner_embedding, max_length=50):
        with torch.no_grad():
            peptide_seq = torch.tensor([self.sep_idx]).reshape(1, -1)
            with torch.no_grad():
                for i in range(max_length):
                    total_length = partner_embedding.shape[1] + peptide_seq.shape[1]
                    padding_mask = torch.zeros(1, total_length, dtype=torch.bool)
                    causal_mask = torch.zeros(total_length, total_length, dtype=torch.bool)
                    logits = self(peptide_seq, partner_embedding, padding_mask, causal_mask)
                    next_token = logits.topk(1, dim=2).indices[0, -1, 0]
                    peptide_seq = torch.cat([peptide_seq, next_token.reshape(1, -1)], dim=1)
                    if next_token.item() == self.eos_idx:
                        break

            return peptide_seq
    
    def training_step(self, batch, batch_idx):
        logits = self(batch['peptide_input'], batch['partner_embedding'], batch['padding_mask'], batch['causal_mask'])
                
        # B x L x C -> B x C x L
        # permute to agree with cross-entropy dims
        logits = logits.permute(0, 2, 1)
        
        loss = F.cross_entropy(logits, batch['peptide_labels'])

        self.log("train_loss", loss, sync_dist=True, batch_size=logits.shape[0])
        
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['peptide_input'], batch['partner_embedding'], batch['padding_mask'], batch['causal_mask'])

        # B x L x C -> B x C x L
        # permute to agree with cross-entropy dims
        logits = logits.permute(0, 2, 1)

        loss = F.cross_entropy(logits, batch['peptide_labels'], ignore_index=-100)

        # -100 labels should be ignored for loss
        loss_indices = batch['peptide_labels'] != -100
        pred = logits.argmax(dim=1)
        accuracy = pred[loss_indices].eq(batch['peptide_labels'][loss_indices]).float().mean()

        self.log("val_loss", loss, sync_dist=True, prog_bar=True, batch_size=logits.shape[0])
        self.log("val_acc", accuracy, sync_dist=True, prog_bar=True, batch_size=logits.shape[0])


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer