import os
import sys

import torch
from advertorch import attacks
from huggingface_hub import hf_hub_download
from torch.nn import CrossEntropyLoss

from data.reader import get_data
from open_flamingo import create_model_and_transforms
from train.train import finetune_clip
from utils.param import parse_args


def model_wrapper(model, part_of_visiox_x, lang_x, attention_mask, max_new_tokens, num_beams, output_scores,
                  return_dict_in_generate):
    def generate(one_img):
        vision_x = torch.concat((part_of_visiox_x, one_img.unsqueeze(0).unsqueeze(0)), dim=1)
        out = model(vision_x=vision_x, lang_x=lang_x,
                    attention_mask=attention_mask)
        print(out)
        return out.logits[:, -1, :]
        # out = model.generate(vision_x=vision_x, lang_x=lang_x,
        #                 attention_mask=attention_mask, max_new_tokens=max_new_tokens,
        #                 num_beams=num_beams, output_scores=output_scores, return_dict_in_generate=return_dict_in_generate)
        # print(torch.stack(list(out.scores), dim=0).squeeze(1).shape)
        # print(torch.stack(list(out.scores), dim=0).squeeze(1).requires_grad)
        # return torch.stack(list(out.scores), dim=0).squeeze(1)

    return generate


if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
        cache_dir=os.getcwd()
    )
    model.requires_grad_(True)

    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    print(f'The model is loaded successfully!')
    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
    data = get_data(args,
                    (image_processor, image_processor),
                    epoch=0,
                    tokenizer=tokenizer,
                    )
    print('The data is loaded successfully')

    for sample in data['train'].dataloader:
        print(len(sample))
        print(sample[0].shape)
        print(len(sample[1]))
        print(sample[1][0])
        print(sample[1][0].ids)
        break
    finetune_clip(model, data, args)

    # lang_x = tokenizer(
    #     ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
    #     return_tensors="pt",
    # )

    # model_generate = model_wrapper(model, part_of_visiox_x=vision_x[:, :2, :, :, :, :],
    #                                lang_x=lang_x["input_ids"], attention_mask=lang_x["attention_mask"],
    #                                max_new_tokens=20, num_beams=1, output_scores=True, return_dict_in_generate=True)
    # attack = attacks.LinfPGDAttack(model_generate, loss_fn=CrossEntropyLoss(), nb_iter=1, eps=4 / 255, eps_iter=2 / 255,
    #                                clip_min=0, clip_max=1, targeted=False)
    #
    # label = torch.tensor([273, 247, 14664, 292, 2829, 15])
    #
    # adv = attack(vision_x[:, -1, :, :, :, :].squeeze(0), label[:1])
    #
    # out = model.generate(vision_x=adv, lang_x=lang_x["input_ids"], attention_mask=lang_x["attention_mask"],
    #                      max_new_tokens=20, num_beams=1, output_scores=True, return_dict_in_generate=True)
    #
    # loss = CrossEntropyLoss()(torch.stack(list(out.scores), dim=0).squeeze(1), label)
    # print(loss)
