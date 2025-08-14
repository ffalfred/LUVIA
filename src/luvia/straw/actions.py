from LUVIA.src.luvia.straw.model.model import ImageToText
from torch import nn
import torch
from tqdm import tqdm

class NeuralActions:

    @staticmethod
    def train_and_validate(model, train_loader, val_loader, optimizer, vocab, pad_idx,
                            device='cuda', num_epochs=10, max_len=21):
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        inv_vocab = {v: k for k, v in vocab.items()}

        loss_train = []
        loss_val = []

        for epoch in range(1, num_epochs + 1):
            # === TRAINING ===
            model.train()
            total_train_loss = 0

            for batch_idx, (images, transcr, split_transcr, captions, len_transcr) in tqdm(enumerate(train_loader)):
                images = images.to(device)
                captions = captions.to(device)

                optimizer.zero_grad()

                outputs = model(images, captions[:, :-1])  # input: all but last
                targets = captions[:, 1:]                  # target: all but first

                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f"[Epoch {epoch}] Batch {batch_idx} | Train Loss: {loss.item():.4f}")

            avg_train_loss = total_train_loss / len(train_loader)
            loss_train.append(avg_train_loss)
            print(f"[Epoch {epoch}] Avg Train Loss: {avg_train_loss:.4f}")

            # === VALIDATION ===
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch_idx, (images, transcr, split_transcr, captions, len_transcr) in tqdm(enumerate(val_loader)):
                    images = images.to(device)
                    captions = captions.to(device)
                    outputs = model(images, captions[:, :-1])
                    targets = captions[:, 1:]

                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                    total_val_loss += loss.item()

                    # Show a few predictions
                    if batch_idx == 0:
                        for i in range(min(3, images.size(0))):
                            prediction = model.infer(image=images[i], start_token=vocab['<START>'],
                                                        end_token=vocab['<END>'], beam_width=3, max_len=max_len, length_norm=True,
                                                        mode="beam")
                            decoded = ''.join([inv_vocab[idx.item()] for idx in prediction[0]])
                            target = ''.join([inv_vocab[idx.item()] for idx in captions[i] if idx.item() != pad_idx])
                            #img_io.imshow(images[i].cpu().permute(1,2,0).numpy())
                            #plt.show()
                            print(f"Target:    {target}")
                            print(f"Predicted: {decoded}")
                            print("-" * 40)

            avg_val_loss = total_val_loss / len(val_loader)
            loss_val.append(avg_val_loss)
            print(f"[Epoch {epoch}] Avg Val Loss: {avg_val_loss:.4f}")
        return loss_train, loss_val

if __name__== "__main__":
    from LUVIA.src.luvia.straw.utils.data_utils import Shorthand_Dataset
    from torch.utils.data import DataLoader
    import torch

    aug_transforms = Shorthand_Dataset.augmentation_functions()
    train_set = Shorthand_Dataset(basefolder="../../../../data/gregg_definitive/", subset='train', 
                                    max_length = 21, rnd_subset = False, transforms=aug_transforms)
    val_set = Shorthand_Dataset(basefolder="../../../../data/gregg_definitive/", subset='val', 
                                    max_length = 21, rnd_subset = True)
    classes = train_set.character_classes

    train_loader = DataLoader(train_set, batch_size=64,
                          shuffle=True, num_workers=8, collate_fn=Shorthand_Dataset.pad_collate)
    val_loader = DataLoader(val_set, batch_size=64,
                            shuffle=False, num_workers=8, collate_fn=Shorthand_Dataset.pad_collate)
    net = ImageToText(len(classes))
    net = net.to('cuda')

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    NeuralActions.train_and_validate(model=net, train_loader=train_loader, val_loader=val_loader,
                        optimizer=torch.optim.Adam(net.parameters(), lr=0.001),
                        vocab=train_set.char_to_num, pad_idx=train_set.char_to_num['<PAD>'], device='cuda', num_epochs=10)

