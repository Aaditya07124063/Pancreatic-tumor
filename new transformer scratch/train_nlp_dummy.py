import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models import Transformer

class CopyDataset(Dataset):
    """
    Toy dataset where the target sequence is the reverse of the source sequence.
    This simulates a translation-like task (reversal) to train the model.
    """
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        src_tokens = torch.randint(3, self.vocab_size, (self.seq_len,))
        tgt_tokens_rev = torch.flip(src_tokens, dims=[0])
        
        sos = torch.tensor([1])
        eos = torch.tensor([2])
        
        tgt_in = torch.cat([sos, tgt_tokens_rev])
        tgt_out = torch.cat([tgt_tokens_rev, eos])
        
        return src_tokens, tgt_in, tgt_out


def main():
    print("=" * 60)
    VOCAB_SIZE = 20
    SEQ_LEN = 8
    BATCH_SIZE = 16
    EPOCHS = 15
    LR = 5e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Training on device: {device}")
    
    dataset = CopyDataset(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, num_samples=1000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_len=100,
        dropout=0.1
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    print("\nStarting Copy/Reversal Task Training...")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            
            optimizer.zero_grad()
            outputs = model(src, tgt_in, src_pad_idx=0, tgt_pad_idx=0)
            
            outputs = outputs.view(-1, VOCAB_SIZE)
            targets = tgt_out.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * src.size(0)
            
            preds = outputs.argmax(dim=-1)
            non_pad_mask = (targets != 0)
            correct += (preds[non_pad_mask] == targets[non_pad_mask]).sum().item()
            total += non_pad_mask.sum().item()
            
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {epoch_loss / len(dataset):.4f} | Token Accuracy: {correct / total * 100:.2f}%")
        
    model.eval()
    with torch.no_grad():
        print("\n" + "=" * 60)
        print("  Evaluating Reversal Inference  ")
        print("=" * 60)
        
        test_src = torch.randint(3, VOCAB_SIZE, (1, SEQ_LEN)).to(device)
        print(f"Input source sequence:  {test_src[0].cpu().tolist()}")
        print(f"Expected target sequence: {test_src[0].cpu().flip(dims=[0]).tolist()}")
        
        tgt_in = torch.tensor([[1]], device=device)
        
        for _ in range(SEQ_LEN + 1):
            outputs = model(test_src, tgt_in, src_pad_idx=0, tgt_pad_idx=0)
            next_token = outputs[:, -1, :].argmax(dim=-1).unsqueeze(1)
            tgt_in = torch.cat([tgt_in, next_token], dim=1)
            
            if next_token.item() == 2:
                break
                
        pred_sequence = tgt_in[0, 1:].cpu().tolist()
        if pred_sequence[-1] == 2:
            pred_sequence = pred_sequence[:-1]
            
        print(f"Model output sequence:   {pred_sequence}")
        success = (pred_sequence == test_src[0].cpu().flip(dims=[0]).tolist())
        print(f"Status: {'SUCCESS ✓' if success else 'FAILURE ✗'}")
        print("=" * 60)

if __name__ == "__main__":
    main()
