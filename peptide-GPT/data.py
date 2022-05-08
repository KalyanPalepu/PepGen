import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import pandas as pd
import pickle
from math import floor
import sys
sys.path.append("/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/esm/")
import esm
import ast

class PartnerPeptideDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, partner_embedding_dict, max_seq_length=1024, max_peptide_length=50):

        self.dataframe = dataframe[dataframe["partner_chain"].isin(partner_embedding_dict.keys())]

        lengths = self.dataframe['partner_seq'].str.len() + self.dataframe['peptide_seq'].str.len()
        self.dataframe = self.dataframe[lengths < max_seq_length - 4 - max_peptide_length] # leave space for bos, sep, eos and peptide
        self.dataframe = self.dataframe.reset_index()
        print(f"Loaded {len(self.dataframe)} sequences")

        self.partner_embedding_dict = partner_embedding_dict

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Select row
        row = self.dataframe.iloc[index]
        pdb_id = row["pdb_id"]
        partner_chain = row["partner_chain"]
        
        return_dict = {
            "pdb_id": pdb_id,
            "peptide_seq": row["peptide_seq"],
            "partner_embedding": self.partner_embedding_dict[partner_chain],
            "binding_region_indices": row['binding_region_indices']
        }            

        return return_dict

class PartnerPeptideCollator:
    def __init__(self, alphabet, use_interface=False):
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.sep_idx = self.alphabet.get_idx("<sep>")
        self.eos_idx = self.alphabet.get_idx("<eos>")
        self.pad_idx = self.alphabet.get_idx("<pad>")

        self.use_interface = use_interface
    
    def __call__(self, raw_batch):
        n_seq = len(raw_batch)
        batch = {}
        _, _, peptide_tokens = self.batch_converter([(d['pdb_id'], d['peptide_seq']) for d in raw_batch])
        peptide_tokens = torch.cat([torch.full((n_seq, 1), self.sep_idx), peptide_tokens], dim=1)
        batch['peptide_input'] = peptide_tokens[:, :-1]
        batch['peptide_labels'] = peptide_tokens[:, 1:].clone()
        if self.use_interface:
            partner_embeddings = []
            for d in raw_batch:
                interface_row = torch.zeros(d['partner_embedding'].shape[1], 1)
                if isinstance(d['binding_region_indices'], str):
                    indices = ast.literal_eval(d['binding_region_indices'])
                else:
                    indices = d['binding_region_indices']

                for i in indices:
                    interface_row[int(i), 0] = 1
                partner_embeddings.append(torch.cat([d['partner_embedding'][0, :, :], interface_row], dim=1))
        else:
            partner_embeddings = [d['partner_embedding'][0, :, :] for d in raw_batch]
        
        batch['partner_embedding'] = pad_sequence(
            partner_embeddings,
            batch_first=True,
            padding_value=0
        )
        
        partner_raw_lengths = [d['partner_embedding'].shape[1] for d in raw_batch]
         
        batch['padding_mask'] = self.get_padding_mask(
            batch['partner_embedding'].shape[1],
            partner_raw_lengths,
            batch['peptide_input']
        )

        batch['causal_mask'] = self.get_causal_mask(
            batch['partner_embedding'].shape[1],
            batch['peptide_input'].shape[1]
        )

        # don't do loss on padding tokens
        batch['peptide_labels'][batch['peptide_labels'] == self.pad_idx] = -100

        return batch
        
    def get_padding_mask(self, partner_batch_length, partner_raw_lengths, peptide_input):
        batch_size = peptide_input.shape[0]
        peptide_length = peptide_input.shape[1]
        total_length = partner_batch_length + peptide_length
                
        mask = torch.ones(batch_size, total_length, dtype=torch.bool)
        
        mask[:, -peptide_length:] = peptide_input == self.pad_idx
        for i, length in enumerate(partner_raw_lengths):
            mask[i, :length] = False
    
    def get_causal_mask(self, partner_batch_length, peptide_length):
        total_length = partner_batch_length + peptide_length
        mask = torch.ones(total_length, total_length, dtype=torch.bool)
        mask[:, :partner_batch_length] = False
        mask[-peptide_length:, -peptide_length:] = ~torch.tril(torch.ones(peptide_length, peptide_length, dtype=torch.bool))
        return mask
        


class PartnerPeptideDataModule(pl.LightningDataModule):
    def __init__(self,
                 regions_csv_path,
                 partner_embeddings_path,
                 train_frac=0.8,
                 test_frac=0.1,
                 val_frac=0.1,
                 batch_size=4,
                 random_seed=42,
                 max_seq_length=1024,
                 max_peptide_length=50,
                 use_interface=False):
        super().__init__()
        
        self.regions_csv_path = regions_csv_path
        self.partner_embeddings_path = partner_embeddings_path
        self.train_frac = train_frac
        self.test_frac = test_frac
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.alphabet = self.get_alphabet()
        self.max_seq_length = max_seq_length
        self.max_peptide_length = max_peptide_length
        self.use_interface = use_interface

    def setup(self, stage):
        regions = pd.read_csv(self.regions_csv_path)
        with open(self.partner_embeddings_path, "rb") as f:
            embeddings = pickle.load(f)

        train_df = regions[regions['split'] == 'train']
        test_df = regions[regions['split'] == 'test']
        val_df = regions[regions['split'] == 'val']

        self.train_dataset = PartnerPeptideDataset(train_df, 
                                                  embeddings, 
                                                  max_seq_length=self.max_seq_length, 
                                                  max_peptide_length=self.max_peptide_length)
        self.test_dataset = PartnerPeptideDataset(test_df, 
                                                  embeddings, 
                                                  max_seq_length=self.max_seq_length, 
                                                  max_peptide_length=self.max_peptide_length)
        self.val_dataset = PartnerPeptideDataset(val_df, 
                                                 embeddings, 
                                                 max_seq_length=self.max_seq_length, 
                                                 max_peptide_length=self.max_peptide_length)
        
        self.collator = PartnerPeptideCollator(self.alphabet, use_interface=self.use_interface)
        
    def get_alphabet(self):
        alphabet = esm.Alphabet.from_architecture("msa_transformer")
        alphabet.prepend_bos = False
        alphabet.append_eos = True
        alphabet.use_msa = False
        return alphabet
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collator, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collator, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collator, num_workers=0)