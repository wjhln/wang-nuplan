from torch import nn

def build_mlp(c_in, channels, norm=None, activation="relu"):
    norm_layer = None
    if norm is not None:
        if norm == "bn":
            norm_layer = nn.BatchNorm1d
        elif norm == "ln":
            norm_layer = nn.LayerNorm
        else:
            raise NotImplementedError(f"Normalization {norm} is not implemented.")

    if activation == "relu":
        activation_layer = nn.ReLU
    elif activation == "gelu":
        activation_layer = nn.GELU
    else:
        raise NotImplementedError(f"Activation {activation} is not implemented.")

    layers = []
    num_layers = len(channels)
    for k in range(num_layers):
        if k == num_layers - 1:
            layers.append(nn.Linear(c_in, channels[k], bias=True))
        else:
            if norm is None:
                layers.extend([nn.Linear(c_in, channels[k], bias=True), activation_layer()])
            else:
                layers.extend(
                    [
                        nn.Linear(c_in, channels[k], bias=False),
                        norm_layer(channels[k]),
                        activation_layer(),
                    ]
                )
            c_in = channels[k]
    return nn.Sequential(*layers)
