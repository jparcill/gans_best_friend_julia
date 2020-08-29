using Flux
using Flux.Optimise: update!

generator = Chain(
    ConvTranspose((4,4), 100=>512; stride=1, pad=0),
    BatchNorm(512, leakyrelu),
    ConvTranspose((4,4), 512=>256; stride=2, pad=1),
    BatchNorm(256, leakyrelu),
    ConvTranspose((4,4), 256=>128; stride=2, pad=1),
    BatchNorm(128, leakyrelu),
    ConvTranspose((4,4), 128=>64; stride=2, pad=1),
    BatchNorm(64, leakyrelu),
    ConvTranspose((4,4), 64=>1, tanh; stride=2, pad=1)
) |> gpu

discriminator = Chain(
    Conv((4,4), 1=>64; stride=1, pad=0),
    BatchNorm(64, leakyrelu),
    Conv((4,4), 64=>128; stride=1, pad=0),
    BatchNorm(128, leakyrelu),
    Conv((4,4), 128=>256; stride=1, pad=0),
    BatchNorm(256, leakyrelu),
    Conv((4,4), 256=>512, sigmoid; stride=1, pad=0),
    BatchNorm(512, sigmoid),
) |> gpu

x = rand(1,1,100,100) |> gpu
data = generator(x)
data2 = discriminator(data)
