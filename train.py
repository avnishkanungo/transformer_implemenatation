import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard.writer import SummaryWriter 

from config import get_config, latest_weights_file_path, get_weights_file_path
from dataset import BilingualDataset, causal_mask
from model import buildTransformer
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import Lowercase, StripAccents
from tokenizers import normalizers
from pathlib import Path

import torchmetrics

def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_token = tokenizer_src.token_to_id('[SOS]')
    eos_token = tokenizer_src.token_to_id('[EOS]')
    
    #pre-comute the encoder output
    encoder_output = model.encode(src, src_mask)
    #initialize the decoder with the start of sentence token
    decoder_input = torch.empty(1,1).fill_(sos_token).type_as(src).to(device)

    while True:
        if decoder_input.size(1) > max_len:
            break

        # mask so that the decoder does not look at the future output tokens
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src).to(device)

        decoder_output = model.decode(decoder_input,encoder_output,src_mask,decoder_mask)

        #gets next token
        prob_dist = model.projection(decoder_output[:,-1])

        #get token with the highest probabliltiy
        _, next_token = torch.max(prob_dist, dim=1)

        #add the next token to the sequence
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1,1).type_as(src).fill_(next_token.item()).to(device)
            ],
            dim=1
        )       

        #if the next token is the end of sentence token, then stop
        if next_token == eos_token:
            break

    return decoder_input.squeeze(0)



def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    #size of the control window(default value for now)
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds: # validation ds will have a batch size of 1
            count+=1
            encoder_input = batch['encoder_input'].to(device) # (Batch, Seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (Batch,1,1 Seq_len)

            assert encoder_input.size(0) == 1, "Batch Size for vaidation dataset needs to be 1"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            source_texts.append(src_text)
            expected.append(tgt_text)
            predicted.append(model_output_text)

            print_msg("="*console_width)
            print_msg(f"Source: {src_text}")
            print_msg(f"Target: {tgt_text}")
            print_msg(f"Predicted: {model_output_text}")
            print_msg("="*console_width)

            if count == num_examples:
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size = 512-len(special_tokens))
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    # ds_raw = ds_raw.with_format("torch")
    tokenizer_src = get_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # train_ds_size = int(0.9 * len(list(ds_raw)))
    # val_ds_size = len(list(ds_raw)) - train_ds_size
    # train_test_split = ds_raw.train_test_split(test_size=0.1)
    
    # train_ds_raw = train_test_split['train']
    # val_ds_raw = train_test_split['test']

    # ds = BilingualDataset(ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"max_len_src: {max_len_src}")
    print(f"max_len_tgt: {max_len_tgt}")

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader,tokenizer_src, tokenizer_tgt

def get_model(config,vocab_size_src, vocab_size_tgt):
    model = buildTransformer(vocab_size_src, vocab_size_tgt,config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    # if (device == 'cuda'):
    #     print(f"Device name: {torch.cuda.get_device_name(device.index())}")
    #     print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    # elif (device == 'mps'):
    #     print(f"Device name: <mps>")
    # else:
    #     print("NOTE: If you have a GPU, consider using it for training.")
    #     print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
    #     print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)
    
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader,tokenizer_src, tokenizer_tgt = get_dataset(config)
    print(f"source:{tokenizer_src.get_vocab_size()}, target: {tokenizer_tgt.get_vocab_size()}")
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    #tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        # model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch']+1
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device) # (Batch, Seq_len)
            decoder_input = batch['decoder_input'].to(device) # (Batch, Seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (Batch,1,1 Seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (Batch,1,seq_len, Seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) # (Batch, Seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (Batch, Seq_len, d_model)
            proj_output = model.project(decoder_output) # (Batch, Seq_len, tgt_vocab_size)
            label = batch['label'].to(device) # (Batch, Seq_len)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss":f"{loss.item():6.3f}"})

            # Logging on tensorboard
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.flush()

            #Backpropgation
            loss.backward()


            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


            global_step += 1
        
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer) 

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    # warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)









