import torch
from model import PepGenGPT
from data import PartnerPeptideDataModule
import os
import pytorch_lightning as pl
import argparse
import torch.nn.functional as F
from Bio import SeqIO
import sys
sys.path.append("/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/esm")
import esm
import itertools

#next 3 fns from https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename):
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence):
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename, nseq):
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict peptide for a given protein')
    parser.add_argument('--partner_seq_file', dest='partner_seq_file', required=True, help="path to fasta file with partner sequence")
    parser.add_argument('--msa_dir', dest="msa_dir", default="msa_output")

    # hhblits arguments
    parser.add_argument('--hhblits_path', dest='hhblits_path', default="hhblits")
    parser.add_argument('--database_path', dest='database_path', help="database used for hhblits")
    parser.add_argument('--n_cpu', dest='n_cpu', type=int, help="number of cpus used by hhblits")
    parser.add_argument('--mact', dest='mact', default=0.35, type=float, help="hhblits mact argument")
    parser.add_argument('--e_value', dest='e_value', default=0.001, type=float, help="hhblits e value argument")
    parser.add_argument('--n_iters', dest='n_iters', default=1, type=float, help="number of hhblits iterations to run")

    # esm
    parser.add_argument('--esm_weights_path', dest="esm_weights_path")

    # model args
    parser.add_argument('--esm_token_embedding_path', dest='esm_token_embedding_path', default="esm_token_embedding.pt")
    parser.add_argument('--max_seq_length', dest='max_seq_length', type=int, default=1024)
    parser.add_argument('--max_peptide_length', dest='max_peptide_length', type=int, default=50)
    parser.add_argument('--esm_dim', dest='esm_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', dest="hidden_dim", type=int, default=512)
    parser.add_argument('--num_heads', dest="num_heads", type=int, default=4)
    parser.add_argument('--num_layers', dest="num_layers", type=int, default=4)
    parser.add_argument('--dim_feedforward', dest="dim_feedforward", type=int, default=64)
    parser.add_argument('--layer_norm_eps', dest="layer_norm_eps", type=float, default=1e-5)
    parser.add_argument('--dropout', dest="dropout", type=float, default=1e-1)
    parser.add_argument('--use_esm_token_embedding', dest='use_esm_token_embedding', action='store_true')
    parser.add_argument('--positional_embed_peptide_only', dest='positional_embed_peptide_only', action='store_true')
    parser.add_argument('--learnable_positional_embedding', dest='learnable_positional_embedding', action='store_true')
    parser.add_argument('--norm_first', dest='norm_first', action='store_true')

    parser.add_argument('--checkpoint', dest="checkpoint", default="/home/gridsan/kalyanpa/DNAInteract_shared/kalyan/pepgen/optuna_models/optuna_3/24/lightning_logs/version_12816901/checkpoints/epoch=99-step=47899.ckpt")

    parser.add_argument('--submit', dest='submit', action='store_true')

    args = parser.parse_args()
    pepgen_config = vars(args)

    os.makedirs(args.workspace_dir, exists=True)

    for seq_record in SeqIO.parse(args.partner_seq_file, "fasta"):
        name = seq_record.id
        seq = seq_record.seq

    # required config but used for prediction purposes
    pepgen_config['lr'] = 1 
    pepgen_config['layer_norm_eps'] = 1

    msa_output = f"{os.path.join(args.msa_dir, name)}.a3m"

    cmd = f"""
    {args.hhblits_path} \
    -cpu {args.n_cpu} \
    -mact {args.mact} \
    -e {args.e_value} \
    -i {args.partner_seq_file} \
    -d {args.database_path} \
    -oa3m {msa_output} \
    -n {args.n_iters}
    """

    os.system(cmd)

    model_data = torch.load(args.esm_weights_path, map_location="cpu")

    esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet_core(model_data, None)
    esm_model = esm_model.eval()
    esm_batch_converter = esm_alphabet.get_batch_converter()

    msa = read_msa(msa_output, 64)
    esm_batch_labels, esm_batch_strs, esm_batch_tokens = esm_batch_converter(msa)

    with torch.no_grad():
        outputs = esm_model(esm_batch_tokens, repr_layers=[12], return_contacts=False)

    # just representation for the main sequence
    embedding = outputs['representations'][12][:, 0, :, :]

    if args.use_esm_token_embedding:
        pepgen_config['esm_token_embedding'] = torch.load(args.esm_token_embedding_path)
    else:
        pepgen_config['esm_token_embedding'] = None

    




    


