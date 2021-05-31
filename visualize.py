import os
import json
import uuid

import torch
import pandas as pd

from config import get_args
from prepare import load_sample
from networks import load_model_and_tokenizer


def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError(
                "The attention tensor does not have the correct number of dimensions. Make sure you set "
                "output_attentions=True when initializing your model."
            )
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def format_special_chars(tokens):
    return [t.replace("Ġ", " ").replace("▁", " ").replace("</w>", "") for t in tokens]


def load_data_by_class_label(args):
    train_path = os.path.join(args.data_dir, "train.tsv")
    train_df = pd.read_csv(train_path, delimiter="\t", header=None)
    train_df = train_df.groupby(8).sample(1)

    return train_df


def head_view(attention, tokens, pretty_tokens=True, layer=None, heads=None):
    """Render head view
    Args:
    attention: list of ``torch.FloatTensor``(one for each layer) of shape
        ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
    tokens: list of tokens
    prettify_tokens: indicates whether to remove special characters in wordpieces, e.g. Ġ
    layer: index of layer to show in visualization when first loads. If non specified, defaults to layer 0.
    heads: indices of heads to show in visualization when first loads. If non specified, defaults to all."""

    vis_id = "bertviz-%s" % (uuid.uuid4().hex)
    vis_html = """
            <div id='%s'>
              <span style="user-select:none">
                Layer: <select id="layer"></select>
              </span>
              <div id='vis'></div> 
            </div>
    """ % (
        vis_id
    )

    if pretty_tokens:
        tokens = format_special_chars(tokens)

    attn = format_attention(attention)
    attn_data = {
        "all": {"attn": attn.tolist(), "left_text": tokens, "right_text": tokens}
    }

    params = {
        "attention": attn_data,
        "default_filter": "all",
        "root_div_id": vis_id,
        "layer": layer,
        "heads": heads,
    }

    attn_seq_len = len(attn_data["all"]["attn"][0][0])

    if attn_seq_len != len(tokens):
        raise ValueError(
            f"Attention has {attn_seq_len} positions, while number of tokens is {len(tokens)}"
        )

    vis_js = (
        open(os.path.join("./", "head_view.js"))
        .read()
        .replace("PYTHON_PARAMS", json.dumps(params))
    )

    template = """<!DOCTYPE html>
<html>
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>
</head>
<body>
    {}
    <script type="text/javascript">
        {}
    </script>
</body>
</html>
""".format(
        vis_html, vis_js
    )

    with open("test.html", "w") as f:
        f.writelines(template)


def bert_attention(args, model, tokenizer):
    """ model이 attention을 Return 한다고 가정."""
    args = get_args()
    train_df = load_data_by_class_label(args)

    model, tokenizer = load_model_and_tokenizer(args)
    inputs, labels = load_sample(args, tokenizer, is_train=False)

    #  tokens = tokenizer.convert_ids_to_tokens(
    #      inputs["input_ids"].detach().cpu().numpy()[0]
    #  )

    attention = model.backbone.electra(**inputs)[-1]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    print(attention.shape, len(tokens))
    #  head_view(attention, tokens)


if __name__ == "__main__":
    from transformers import AutoConfig

    args = get_args()
    model, tokenizer = load_model_and_tokenizer(args)

    inputs, labels = load_sample(args, tokenizer)

    config = AutoConfig.from_pretrained(args.model_name_or_path, output_attentions=True)
    config.num_labels = 42

    model.backbone.__init__(config=config)
    model.load_state_dict(torch.load("./weights/st00_testmodel_001.pth"))

    model.to(args.device)

    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map["pad_token"])
    idx = inputs["input_ids"][0].tolist().index(pad_id)

    input_ids = inputs["input_ids"][:1, :idx]
    token_type_ids = inputs["token_type_ids"][:1, :idx]

    print(input_ids, token_type_ids)

    outputs = model.backbone(input_ids, token_type_ids=token_type_ids, return_dict=True)

    attention = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    head_view(attention, tokens)
