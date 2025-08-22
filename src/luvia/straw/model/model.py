#from LUVIA.src.luvia.straw.model.encoder import CNNEncoder
#from LUVIA.src.luvia.straw.model.decoder import LSTMDecoder
from luvia.straw.model.encoder import CNNEncoder
from luvia.straw.model.decoder import LSTMDecoder
from torch import nn

# Full Model
class ImageToText(nn.Module):

    def __init__(self, vocab_size, encoded_dim=256, hidden_dim=512):
        super(ImageToText, self).__init__()
        self.encoder = CNNEncoder(encoded_dim)
        self.decoder = LSTMDecoder(encoded_dim, hidden_dim, vocab_size)

    def forward(self, images, captions):
        features, act1, act2 = self.encoder(images)
        return self.decoder(features, captions), act1, act2

    def infer(self, image, start_token, end_token, max_len=21, mode="vanilla", beam_width=3, length_norm=True,
                num_groups=3, diversity_strength=0.5, top_k=0, top_p=0.9, temperature=1.0, k=1):
        features, act1, act2 = self.encoder(image.unsqueeze(0))
        if mode == "vanilla":
            infer_result = self.infer_vanilla(features, start_token, end_token, max_len=max_len)
        elif mode == "beam":
            infer_result = self.infer_beam(features, start_token, end_token, beam_width=beam_width,
                                            max_len=max_len, length_norm=length_norm, k=k)
        elif mode == "diverse_beam":
            infer_result = self.infer_diverse_beam(features, start_token, end_token, beam_width=beam_width,
                                                    max_len=max_len, length_norm=length_norm,  num_groups=num_groups,
                                                    diversity_strength=diversity_strength, k=k)
        elif mode == "sample":
            print("FEatures", features.shape)
            infer_result = self.infer_sample(features, start_token, end_token, max_len=max_len, top_k=top_k, top_p=top_p,
                                                temperature=temperature, k=k)
        else:
            raise ValueError("Mode {} is not coded yet".format(mode))
        return infer_result, act1, act2

    def infer_vanilla(self, features, start_token, end_token, max_len=21):
        return self.decoder.generate(features, start_token, end_token, max_len)

    def infer_beam(self, features, start_token, end_token, beam_width=3, max_len=21, length_norm=True, k=1):
        return self.decoder.generate_beam_search(features, start_token, end_token, beam_width, max_len, length_norm, k)

    def infer_diverse_beam(self, features, start_token, end_token, beam_width=6, num_groups=3, diversity_strength=0.5,
                            max_len=21, length_norm=True, k=1):
        return self.decoder.generate_diverse_beam_search(features, start_token, end_token, beam_width, num_groups,
                                                            diversity_strength, max_len, length_norm, k)

    def infer_sample(self, features, start_token, end_token, max_len=21, top_k=0, top_p=0.9, temperature=1.0, k=1):
        return self.decoder.generate_topk_topp_sampling(features, start_token, end_token, max_len, top_k, top_p, temperature, k)
