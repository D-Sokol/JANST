#!/usr/bin/env python3
import torch
from torch.nn.functional import mse_loss
from torchvision.io import read_image
from torchvision.models import vgg11  # TODO: what about other models?
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, ToPILImage
from tqdm import trange
import matplotlib.pyplot as plt


def get_cnn(device='cpu'):
    cnn = vgg11(pretrained=True).features.to(device)
    assert isinstance(cnn, torch.nn.Sequential)
    for layer in cnn:
        if isinstance(layer, torch.nn.ReLU):
            layer.inplace = False
    return cnn


def load(path):
    return read_image(path).to(torch.float) / 255


def save(image, path):
    image = image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    plt.imsave(path, image)


def extract_features(model, image, content_layers, style_layers):
    x = image if image.ndim == 4 else image[None]
    content_features = []
    style_features = []
    for layer_id, layer in enumerate(model):
        x = layer(x)
        if layer_id in content_layers:
            content_features.append(x)
        if layer_id in style_layers:
            assert x.ndim == 4
            features = x.flatten(start_dim=2).flatten(end_dim=1)
            gram = features @ features.T
            gram /= x.numel()  # TODO: check against the article
            style_features.append(gram)
    return content_features, style_features


def losses(model, generated_img, content_features, style_features, content_layers, style_layers):
    assert len(content_layers) == len(content_features)
    assert len(style_layers) == len(style_features)
    assert generated_img.ndim == 4
    x = generated_img

    content_features_gen, style_features_gen = extract_features(model, x, content_layers, style_layers)
    content_loss = sum(mse_loss(*pair) for pair in zip(content_features_gen, content_features)) / len(content_features)
    style_loss = sum(mse_loss(*pair) for pair in zip(style_features_gen, style_features)) / len(style_features)
    return content_loss, style_loss


def process(content_image, style_image, cnn=cnn, device=device,
            # TODO: better labelling
            content_layers=(13,), style_layers=(0, 3, 6, 11, 16),
            style_weight=1e4, lr=1e-1,
            n_epoch=10000, print_every=None, show_progress=False):

    # Recommended normalization parameters for VGG network
    normalize_network = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # Store parameters of style image to unnormalize the result back
    style_mean = style_image.mean((1,2))[..., None, None].to(device)
    style_std = style_image.std((1,2))[..., None, None].to(device)

    preprocess = Compose([
        ToPILImage(),
        #Resize(512),
        ToTensor(),
        normalize_network
    ])

    content_image = preprocess(content_image).to(device)[None]
    style_image = preprocess(style_image).to(device)[None]

    with torch.no_grad():
        content_features = extract_features(cnn, content_image, content_layers, style_layers)[0]
        style_features = extract_features(cnn, style_image, content_layers, style_layers)[1]

    gen = torch.randn_like(content_image, requires_grad=True)
    opt = torch.optim.Adam((gen,), lr=lr)

    for epoch in (trange(n_epoch) if show_progress else range(n_epoch)):
        c_loss, s_loss = losses(cnn, gen, content_features, style_features, content_layers, style_layers)
        (c_loss + style_weight * s_loss).backward()
        assert gen.grad is not None
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            gen.clamp_(0, 1)
        if print_every is not None and epoch % print_every == 0:
            print(epoch + 1, c_loss.item(), s_loss.item(), sep='\t')

    # Returns both original reconstructed and unnormalized reconstructed images
    return (gen, style_mean + style_std * gen)


def shortcut(content_path, style_path):
    img1, img2 = process(
        load(content_path), load(style_path),
        get_cnn(), torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        print_every=500, show_progress=False
    )
    save(img1, 'generated1.jpg')
    save(img2, 'generated2.jpg')


if __name__ == '__main__':
    import sys
    shortcut(sys.argv[1], sys.argv[2])


