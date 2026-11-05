for the complex model, we use make an autoencoder out of complex conv kernels and complex activations. here's how:
let's consider input as complex number input = a + ib.
for conv layer, consider a complex kernel = x + iy.
so when you conv these (using * symbol for conv from here),
input * kernel = (a + ib) * (x + iy) = (a*x + i b*x) + (i a*y - b*y)
which is equal to (a*x - b*y) + i (b*x + a*y)
which means that we can represent a complex conv layer using regular nn.Module.
for activation (let's use ReLU here), we can do ReLU(a) + i ReLU(b).
which means that we can have an autoencoder capable to handling complex numbers.