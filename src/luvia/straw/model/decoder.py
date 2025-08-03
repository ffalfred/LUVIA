from torch import nn
import torch
import torch.nn.functional as F



class LSTMDecoder(nn.Module):


    def __init__(self, encoded_dim, hidden_dim, vocab_size, embedding_dim=128, num_layers=1, dropout=0.3):
        
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # New input size: embedding + image features
        self.lstm = nn.LSTM(embedding_dim + encoded_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.init_hidden = nn.Linear(encoded_dim, hidden_dim)

    def forward(self, features, captions):
        embedded = self.embedding(captions)  # (B, T, E)
        embedded = self.dropout(embedded)

        # Repeat image features across time steps
        features_expanded = features.unsqueeze(1).expand(-1, embedded.size(1), -1)  # (B, T, F)
        lstm_input = torch.cat((embedded, features_expanded), dim=2)  # (B, T, E+F)

        h0 = self.init_hidden(features).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        out, _ = self.lstm(lstm_input, (h0, c0))
        out = self.dropout(out)
        return self.fc(out)

    def generate(self, features, start_token, end_token, max_len=20):
        batch_size = features.size(0)
        h = self.init_hidden(features).unsqueeze(0)
        c = torch.zeros_like(h)

        inputs = torch.tensor([[start_token]] * batch_size, device=features.device)
        outputs = []

        for _ in range(max_len):
            embedded = self.embedding(inputs)  # (B, 1, E)
            embedded = self.dropout(embedded)

            features_expanded = features.unsqueeze(1)  # (B, 1, F)
            lstm_input = torch.cat((embedded, features_expanded), dim=2)  # (B, 1, E+F)

            out, (h, c) = self.lstm(lstm_input, (h, c))
            out = self.dropout(out)

            logits = self.fc(out.squeeze(1))
            predicted = logits.argmax(1)
            outputs.append(predicted)

            inputs = predicted.unsqueeze(1)
            if (predicted == end_token).all():
                break
        return torch.stack(outputs, dim=1)

    def generate_beam_search(self, features, start_token, end_token, beam_width=3, max_len=20, length_norm=True, k=1):
        device = features.device
        batch_size = features.size(0)
        assert batch_size == 1, "Beam search currently supports batch_size=1"

        h = self.init_hidden(features).unsqueeze(0)
        c = torch.zeros_like(h)

        beams = [([start_token], 0.0, h, c)]  # (tokens, score, h, c)

        for _ in range(max_len):
            new_beams = []
            for tokens, score, h, c in beams:
                if tokens[-1] == end_token:
                    new_beams.append((tokens, score, h, c))
                    continue

                input_token = torch.tensor([[tokens[-1]]], device=device)
                embedded = self.dropout(self.embedding(input_token))

                features_expanded = features.unsqueeze(1)
                lstm_input = torch.cat((embedded, features_expanded), dim=2)

                out, (h_new, c_new) = self.lstm(lstm_input, (h, c))
                out = self.dropout(out)

                logits = self.fc(out.squeeze(1))
                log_probs = torch.log_softmax(logits, dim=1)

                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                for log_prob, idx in zip(topk_log_probs[0], topk_indices[0]):
                    new_tokens = tokens + [idx.item()]
                    new_score = score + log_prob.item()
                    new_beams.append((new_tokens, new_score, h_new, c_new))

            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x[1] / len(x[0]) if length_norm else x[1], reverse=True)[:beam_width]

            # Early stopping if all beams ended
            if all(b[0][-1] == end_token for b in beams):
                break

        return [torch.tensor(tokens[1:], device=device) for tokens, _, _, _ in beams[:k]]

    def generate_diverse_beam_search(self, features, start_token, end_token, beam_width=6, num_groups=3,
                                        diversity_strength=0.5, max_len=20, length_norm=True, k=1):
        device = features.device
        batch_size = features.size(0)
        assert batch_size == 1, "Diverse beam search currently supports batch_size=1"

        h = self.init_hidden(features).unsqueeze(0)
        c = torch.zeros_like(h)

        group_size = beam_width // num_groups
        beams = [[([start_token], 0.0, h, c)] for _ in range(num_groups)]

        for _ in range(max_len):
            new_beams = [[] for _ in range(num_groups)]

            for g in range(num_groups):
                seen_tokens = set()
                for tokens, score, h, c in beams[g]:
                    if tokens[-1] == end_token:
                        new_beams[g].append((tokens, score, h, c))
                        continue

                    input_token = torch.tensor([[tokens[-1]]], device=device)
                    embedded = self.dropout(self.embedding(input_token))
                    features_expanded = features.unsqueeze(1)
                    lstm_input = torch.cat((embedded, features_expanded), dim=2)

                    out, (h_new, c_new) = self.lstm(lstm_input, (h, c))
                    out = self.dropout(out)

                    logits = self.fc(out.squeeze(1))
                    log_probs = torch.log_softmax(logits, dim=1)

                    # Apply diversity penalty
                    for token_id in seen_tokens:
                        log_probs[0, token_id] -= diversity_strength

                    topk_log_probs, topk_indices = torch.topk(log_probs, group_size)

                    for log_prob, idx in zip(topk_log_probs[0], topk_indices[0]):
                        seen_tokens.add(idx.item())
                        new_tokens = tokens + [idx.item()]
                        new_score = score + log_prob.item()
                        new_beams[g].append((new_tokens, new_score, h_new, c_new))

                # Keep top beams per group
                new_beams[g] = sorted(new_beams[g], key=lambda x: x[1] / len(x[0]) if length_norm else x[1],
                                        reverse=True)[:group_size]
            beams = new_beams

            # Early stopping if all beams ended
            if all(all(b[0][-1] == end_token for b in group) for group in beams):
                break


        # Final step: collect top sequences from all groups
        all_sequences = [(tokens, score) for group in beams for tokens, score, _, _ in group]

        # Sort and return top `num_outputs`
        sorted_sequences = sorted(all_sequences, key=lambda x: x[1] / len(x[0]) if length_norm else x[1],
                                    reverse=True)[:k]
        return [torch.tensor(tokens[1:], device=device) for tokens, _ in sorted_sequences]



    def generate_topk_topp_sampling(self, features, start_token, end_token, max_len=20, top_k=0, top_p=0.9,
                                        temperature=1.0, num_samples=5):
         #DOESNT WORK
        import torch.nn.functional as F

        device = features.device
        batch_size = features.size(0)
        assert batch_size == 1, "Sampling currently supports batch_size=1"

        # Ensure features is 2D: (1, encoded_dim)
        features = features.view(1, -1)  # Flatten if needed

        # Prepare hidden and cell states
        h_base = self.init_hidden(features).unsqueeze(0)  # (1, 1, hidden_dim)
        c_base = torch.zeros_like(h_base)

        all_outputs = []

        for _ in range(num_samples):
            h = h_base.clone()
            c = c_base.clone()
            inputs = torch.tensor([[start_token]], device=device)
            outputs = []

            for _ in range(max_len):
                embedded = self.dropout(self.embedding(inputs))  # (1, 1, embedding_dim)
                features_expanded = features.unsqueeze(1)        # (1, 1, encoded_dim)
                lstm_input = torch.cat((embedded, features_expanded), dim=2)  # (1, 1, embedding+encoded_dim)

                out, (h, c) = self.lstm(lstm_input, (h, c))
                out = self.dropout(out)

                logits = self.fc(out.squeeze(1)) / temperature
                probs = F.softmax(logits, dim=-1)

                # Top-k filtering
                if top_k > 0:
                    topk_probs, topk_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(1, topk_indices, topk_probs)
                    probs = probs / probs.sum()

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative_probs > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_probs[mask] = 0.0
                    probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
                    probs = probs / probs.sum()

                next_token = torch.multinomial(probs, num_samples=1)
                outputs.append(next_token.item())

                if next_token.item() == end_token:
                    break

                inputs = next_token.unsqueeze(0)  # (1, 1)

            all_outputs.append(torch.tensor(outputs, device=device))

        return all_outputs


