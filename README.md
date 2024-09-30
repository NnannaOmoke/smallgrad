# Autograd Engine in Rust
Andrej Karpathy, in a video on his youtube channel: [https://www.youtube.com/watch?v=VMj-3S1tku0&t=1741s], built 
an autograd engine called `micrograd`. For educational purposes, I'm translating it to rust, and I hope to extend it
by adding more functionality, i.e. by computing the derivatives of tensors, as opposed to just performing backpropagation
for scalar values.
